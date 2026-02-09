# %%
import pandas as pd
import os
import sys
import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import re
import sys

sys.path.append('..')
from misc import *
from metrics import analyze_asyncgroups
from tqdm import tqdm


def convert_to_simple_format(parsed_list):
    """
    Converts a parsed list structure into a simplified string format for output.

    Args:
        parsed_list (list): List of parsed dicts, with 'type' (sync/async),
         'content', and 'title' for async.

    Returns:
        str: Simplified text representation. Async branches are output with
         a custom <branch> tag.
    """
    output = []
    for item in parsed_list:
        if item["type"] == "sync":
            output.append(tokenizer.decode(item["content"]))
        elif item["type"] == "async":
            async_lines = []
            for async_item in item["content"]:
                _t = tokenizer.decode(async_item["title"])
                _c = tokenizer.decode(async_item["content"][1:-1])
                async_lines.append(f'<branch title="{_t}">{_c}</branch>')
            output.append('\n'.join(async_lines))
    return ''.join(output)


def clear_branch_info(content):
    """
    Removes special async-related markup and forbidden words from the content string.

    Args:
        content (str): Original content string possibly containing special tags.

    Returns:
        str: Cleaned content string with tags and forbidden words removed.
    """

    f = content.find('<|title|>')
    r = content.find('<|para|>')
    while f != -1 and r != -1 and f < r:
        content = content[:f] + content[r + len('<|para|>'):]
        f = content.find('<|title|>')
        r = content.find('<|para|>')
    f = content.find('<|title|>')
    r = content.find('<|/title|>')
    while f != -1 and r != -1 and f < r:
        content = content[:f] + content[r + len('<|/title|>'):]
        f = content.find('<|title|>')
        r = content.find('<|/title|>')
    for fw in forbidden_words:
        content = content.replace(fw, '')
    return content


def make_serial_text(item, cnt):
    content = tokenizer.decode(item["content"])
    if cnt == 1 and content.startswith(':'):
        content = content[1:]
    content = content.replace('<|/para|>', '')
    content = content.replace('<|/branch|>', '')
    content = content.replace('<|branch|>', '')
    content = content.replace('</s>', '')
    f = content.find('<|title|>')
    r = content.find('<|para|>')
    if f != -1 and r != -1 and f < r:
        new_content = content[:f] + content[r + len('<|para|>'):]
        return new_content
    else:
        return content


def convert_to_plain_format(parsed_list):
    """
    Convert parsed result list to a plain text format, removing special tags and async markup.

    Args:
        parsed_list (list): List of parsed dicts containing 
            'type', 'content', and possibly 'title'.

    Returns:
        str: Plain text string with tags and async annotations removed.
    """
    output = []
    cnt = 0
    for item in parsed_list:
        cnt += 1
        if item["type"] == "sync":
            output.append(make_serial_text(item, cnt))
        elif item["type"] == "async":
            async_lines = [f'{tokenizer.decode(async_item["content"][1 + len(async_item["title"]) + 1:-1])}'
                           for async_item in item["content"]]
            async_lines = [clear_branch_info(al) for al in async_lines]
            output.append('\n'.join(async_lines))
    return ''.join(output)


import logging
import shortuuid

forbidden_words = ['<|branch|>', '<|/branch|>', '<|title|>', '<|/title|>', '<|para|>', '<|/para|>', '</s>']

import shortuuid
import argparse


def merge_para_infos(para_infos):
    """
Merge the first two paragraph info items when both are of type "sync".

This function checks the first two entries in `para_infos`. If both have
type == "sync", it concatenates their "content" fields into a single item
and returns a new list with the merged item followed by the remaining items.
If the condition is not met or the list has fewer than two items, the
original list is returned unchanged.

Args:
    para_infos (list[dict]): A list of paragraph info dictionaries, each
        expected to contain at least keys:
        - "type" (str): the paragraph type (e.g., "sync").
        - "content" (str): the paragraph content to be concatenated.

Returns:
    list[dict]: A possibly modified list where the first two "sync" items
    are merged into one, or the original list if no merge applies.
"""


if len(para_infos) > 1:
    if para_infos[0]["type"] == "sync" and para_infos[1]["type"] == "sync":
        new_para_item = {"type": "sync", "content": para_infos[0]["content"] + para_infos[1]["content"]}
        return [new_para_item] + para_infos[2:]
return para_infos


def determine_paramode(args):
    """
    Determine which model mode to use (parallel or raw) and resolve model paths.

    Preference is given to parallel mode if `args.para_model` is provided;
    otherwise, raw mode is selected if `args.raw_model` is provided.
    The function builds the absolute model path from `args.model_dir` and the
    selected model name.

    Args:
        args: An object (e.g., argparse.Namespace) providing:
            - args.model_dir (str): Base directory where models are stored.
            - args.para_model (str | None): Name of the parallel model, if any.
            - args.raw_model (str | None): Name of the raw model, if any.

    Returns:
        tuple:
            - use_raw (bool): True if raw mode is selected.
            - use_parallel (bool): True if parallel mode is selected.
            - model_path (str): Filesystem path to the chosen model.
            - model_name (str): The chosen model's name.

    Raises:
        RuntimeError: If neither `args.para_model` nor `args.raw_model` is provided.
    """
    use_parallel = False
    use_raw = False
    if args.para_model:
        use_parallel = True
        model_path = os.path.join(args.model_dir, args.para_model)
        model_name = args.para_model
    elif args.raw_model:
        use_raw = True
        model_path = os.path.join(args.model_dir, args.raw_model)
        model_name = args.raw_model
    else:
        raise RuntimeError(f'Please pass a value for one of para-mode, raw-model')
    return use_raw, use_parallel, model_path, model_name


def reload_test_data(args, bench_file, detail_save_file, eval_save_file):
    """
    Reload previously saved evaluation artifacts and determine remaining inputs.

    This function reads:
      - the full benchmark input from `bench_file`,
      - previously saved per-question detailed stats from `detail_save_file`,
      - previously saved evaluation results from `eval_save_file`.

    If `args.force` is True, previously saved results are ignored with a warning.
    Otherwise, it reloads what exists and computes `next_data` as the subset of
    input questions not yet present in saved results.

    Args:
        args: An object (e.g., argparse.Namespace) providing:
            - args.force (bool): If True, ignore and overwrite any saved results.
        bench_file (str): Path to the benchmark JSONL input file.
        detail_save_file (str): Path to the JSONL file with detailed per-item stats.
        eval_save_file (str): Path to the JSONL file with saved evaluation results.

    Returns:
        tuple:
            - input_data (list[dict]): All input records loaded from `bench_file`.
            - save_data (list[dict]): Previously saved evaluation results (may be empty).
            - next_data (list[dict]): Inputs not yet evaluated (to be processed next).
            - para_stats_arr_all (list[dict]): Previously saved detailed stats (may be empty).

    Side Effects:
        - Prints warnings and reload summaries to stdout.

    Notes:
        - Expects each record to have a 'question_id' key used for matching.
    """
    save_data = []
    para_stats_arr_all = []
    input_data = read2jsonline(bench_file)
    if args.force:
        print(f"Warning: All saved result will be clean!!!")
    else:
        if os.path.isfile(detail_save_file):
            para_stats_arr_all = read2jsonline(detail_save_file)
            print(f'Reload from {detail_save_file}: [{len(para_stats_arr_all)}/]')
        if os.path.isfile(eval_save_file):
            save_data = read2jsonline(eval_save_file)
        print(f'Reload from {eval_save_file}: [{len(save_data)}/{len(input_data)}]')

    QUESTION_KEY = 'question_id'
    if len(save_data) < len(input_data):
        map_data = {}
        for d in save_data:
            qkey = d[QUESTION_KEY]
            map_data[qkey] = d
        next_data = []
        for d in input_data:
            qkey = d[QUESTION_KEY]
            if qkey not in map_data:
                next_data.append(d)
        del map_data
    else:
        next_data = []
    return input_data, save_data, next_data, para_stats_arr_all


def determine_model(args, model_path, attn_implementation):
    """
    Load and return the appropriate model and tokenizer based on mode selection.

    If `args.para_model` is set, loads a parallel-capable model
    (Qwen2ForParaCausalLM) and its tokenizer. Otherwise, loads a standard
    causal LM (AutoModelForCausalLM) and its tokenizer. Tokenizer padding
    side is set to "left" for generation compatibility.

    Args:
        args: An object (e.g., argparse.Namespace) providing:
            - args.para_model (str | None): If set, selects parallel model path.
        model_path (str): Filesystem path to the model to load.
        attn_implementation (str): Attention backend implementation identifier
            to pass to the model loader (e.g., "sdpa", "eager", "flash_attention_2").

    Returns:
        tuple:
            - model: The parallel model instance if in parallel mode; otherwise None.
            - tokenizer: The tokenizer for the parallel model; otherwise None.
            - normal_model: The standard causal LM if not in parallel mode; otherwise None.
            - normal_tokenizer: The tokenizer for the standard model; otherwise None.

    Notes:
        - Uses torch_dtype="auto", device_map="auto", and trust_remote_code=True.
        - Requires classes/functions: Qwen2ForParaCausalLM, AutoModelForCausalLM,
          AutoTokenizer, and a valid torch installation.
    """
    model, tokenizer, normal_model, normal_tokenizer = None, None, None, None
    if args.para_model:
        model = Qwen2ForParaCausalLM.from_pretrained(model_path,
                                                     torch_dtype="auto",
                                                     device_map="auto",
                                                     attn_implementation=attn_implementation,
                                                     trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer.padding_side = "left"
    else:
        normal_model = AutoModelForCausalLM.from_pretrained(model_path,
                                                            torch_dtype="auto",
                                                            device_map="auto",
                                                            attn_implementation=attn_implementation,
                                                            trust_remote_code=True)
        normal_tokenizer = AutoTokenizer.from_pretrained(model_path)
        normal_tokenizer.padding_side = "left"
    return model, tokenizer, normal_model, normal_tokenizer


def para_infer(para_messages, q, tokenizer, model, para_st_token, para_ed_token):
    normal_text = ''
    para_text = ''
    normal_text = ''
    normal_num_tokens = ''
    normal_token_speed = ''
    normal_time = ''
    para_text = ''
    para_plain_text = ''
    time1 = ''
    time2 = ''
    total_para_time = ''
    para_num_tokens = ''
    para_token_speed = ''
    para_token_speed1 = ''
    para_token_speed2 = ''
    total_para_token_speed = ''
    all_text_tokens = ''
    all_branch_content_tokens = ''
    para_rate = ''
    para_efficient = ''
    num_para = ''
    para_first_token_time = ''
    normal_first_token_time = ''
    para_messages.append({"role": "user", "content": q})
    print(f'==============>CURRENT MESSAGE:\n{para_messages}')
    text = tokenizer.apply_chat_template(
        para_messages,
        tokenize=False,
        add_generation_prompt=True,
        # enable_thinking=True # Switches between thinking and non-thinking modes. Default is True.
    )
    normal_model_inputs = tokenizer([text], return_tensors="pt").to(model.device)['input_ids']
    # print(f'para_normal_model_inputs is 【{normal_model_inputs}】')
    torch.cuda.empty_cache()
    # setup_seed(1030)
    try:
        # print(f'para_input_text is 【{tokenizer.decode(normal_model_inputs[0])}】')
        res = model.generate_with_parallel(
            normal_model_inputs,
            attention_mask=None,
            use_cache=True,
            eos_token_id=tokenizer.eos_token_id,
            para_begin_token_id=para_st_token,  # <branch>
            para_end_token_id=para_ed_token,  # </branch>
            title_start_token=151667,  # <title>
            title_end_token=151668,  # </title>
            stage1_max_new_tokens=2048,
            stage2_max_new_tokens=2048,
            stage3_max_new_tokens=1024,
            sync_start_id=151669,
            sync_end_id=151670,
            # conclustion_ids=tokenizer.encode('<conclusion>'),
            temperature=0.7,
            top_k=20,
            top_p=0.8,
            return_parallel_info=True,
        )
        para_infos = merge_para_infos(res["para_infos"])
        # print(json.dumps(para_infos, ensure_ascii=False, indent=2))
        para_first_token_time = res["first_token_time"]
        para_text = convert_to_simple_format(para_infos)
        para_plain_text = convert_to_plain_format(para_infos)
        print(f'para_text====>【{para_text}】\n\n')
        print(f'plain_text====>【{para_plain_text}】\n\n')
        para_messages.append({"role": "assistant", "content": para_plain_text})
        save_turns.append(para_plain_text)

        time1 = res['seq_time']
        time2 = res['para_time']

        para_stats = analyze_asyncgroups([para_infos], tokenizer)
        all_branch_content_tokens = para_stats["total_branch_content_tokens"]
        all_text_tokens = para_stats["total_text_tokens"]
        all_title_content_tokens = para_stats["total_title_tokens"]
        if time2 > 0.0:
            total_para_time = time1 + time2
            para_num_tokens = all_text_tokens

            para_rate = para_stats['apd']
            para_efficient = para_stats['ape']
            num_para = para_stats["abn"]
            para_token_speed1 = (all_text_tokens - all_branch_content_tokens) / (time1)
            para_token_speed2 = (all_branch_content_tokens) / (time2)
            total_para_token_speed = para_num_tokens / (time1 + time2)
        else:
            para_rate = ''
            para_efficient = ''
            num_para = ''
            para_token_speed1 = all_text_tokens / (time1)
            para_token_speed2 = ''
            para_num_tokens = all_text_tokens
            total_para_token_speed = all_text_tokens / (time1)
            total_para_time = time1
            time2 = ''
        del res
    except Exception as e:
        logging.exception(e)
        print(f'Error json:', json.dumps(d, ensure_ascii=False, indent=2))
        para_text = ''
        para_plain_text = ''
        time1 = ''
        time2 = ''
        total_para_time = ''
        para_num_tokens = ''
        para_token_speed = ''
        para_token_speed1 = ''
        para_token_speed2 = ''
        total_para_token_speed = ''
        all_text_tokens = ''
        all_branch_content_tokens = ''
        para_rate = ''
        para_efficient = ''
        num_para = ''
    return {
        'para_text': para_text,
        'normal_text': '',
        "num_para": num_para,
        'para_rate': para_rate,
        "para_efficient": para_efficient,
        "para_num_tokens": para_num_tokens,
        "total_para_time": total_para_time,
        "para_first_token_time": para_first_token_time,
        "total_para_token_speed": total_para_token_speed,
        "seq_time": time1,
        "seq_speed": para_token_speed1,
        "para_time": time2,
        "para_speed": para_token_speed2,
        "all_text_tokens": all_text_tokens,
        "all_branch_content_tokens": all_branch_content_tokens,
    }


def normal_infer(normal_messages, q, normal_tokenizer, normal_model):
    normal_text = ''
    normal_num_tokens = ''
    normal_token_speed = ''
    normal_time = ''
    normal_first_token_time = ''
    normal_messages.append({"role": "user", "content": q})
    print(f'==============>CURRENT MESSAGE:\n{normal_messages}')
    text = normal_tokenizer.apply_chat_template(
        normal_messages,
        tokenize=False,
        add_generation_prompt=True,
        # enable_thinking=True # Switches between thinking and non-thinking modes. Default is True.
    )
    normal_model_inputs = normal_tokenizer([text], return_tensors="pt").to(normal_model.device)['input_ids']
    # print(f'normal_model_inputs.shape={normal_model_inputs.shape}')
    # print('normal_model_inputs=>', normal_model_inputs)
    torch.cuda.empty_cache()
    # setup_seed(1024)
    st_time = time.time()
    tmp_dict = normal_model.generate(normal_model_inputs,
                                     temperature=0.7, top_k=20, top_p=0.8,
                                     max_new_tokens=1, return_dict_in_generate=True, use_cache=True)
    normal_first_token_time = time.time() - st_time
    generated_sequence = tmp_dict.sequences
    past_key_values = tmp_dict.past_key_values
    try:
        start_time = time.time()
        res = normal_model.generate(input_ids=generated_sequence,
                                    temperature=0.7, top_k=20, top_p=0.8,
                                    max_new_tokens=5096, use_cache=True, past_key_values=past_key_values)
        normal_time = time.time() - start_time

        normal_text = normal_tokenizer.decode(
            res[0, normal_model_inputs.shape[-1]:], skip_special_tokens=True)
        print('normal_text=>', normal_text)

        normal_num_tokens = res.shape[-1] - normal_model_inputs.shape[-1]
        normal_text = normal_text.replace('</s>', '')
        normal_token_speed = normal_num_tokens / (normal_time)

        normal_time += normal_first_token_time
        normal_text = normal_text.replace(MASK, '')
        normal_text = normal_text.replace(STREAM_START, '')
        normal_messages.append({"role": "assistant", "content": normal_text})
        save_turns.append(normal_text)
    except Exception as e:
        print(f'Error is>>>>>>>>>>>: {e}')
        logging.exception(e)
        normal_text = ''
        normal_num_tokens = ''
        normal_token_speed = ''
        normal_time = ''

    return {
        "normal_text": normal_text,
        "para_text": '',
        "normal_num_tokens": normal_num_tokens,
        "total_normal_time": normal_time,
        "normal_first_token_time": normal_first_token_time,
        "normal_token_speed": normal_token_speed,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-b",
        "--bench-name",
        type=str,
        choices=["mt_bench", "vicuna_bench", "rag_bench"],
        default="mt_bench",
        help="The name of the benchmark question set.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite all saved results!",
    )
    parser.add_argument(
        "-r",
        "--run-id",
        default=0,
        type=int,
    )
    parser.add_argument(
        "-s",
        "--base-save-dir",
        default=r'APAR_BENCH_Qwen25-7B_RES',
        type=str,
    )
    parser.add_argument(
        "-md",
        "--model-dir",
        default=r'../train/saves/Qwen25-7B/full/',
        type=str,
    )
    parser.add_argument(
        "-pm",
        "--para-model",
        default=None,
        type=str,
    )
    parser.add_argument(
        "-rm",
        "--raw-model",
        default=None,
        type=str,
    )

    args = parser.parse_args()
    use_raw, use_parallel, model_path, model_name = determine_paramode(args)
    print('model_path', model_path)
    print('para_model', args.para_model)
    print('raw_model', args.raw_model)

    from modeling_qwen2_aspd import Qwen2ForParaCausalLM

    attn_implementation = 'sdpa'

    bench_file = fr'benchmarks/{args.bench_name}/question.jsonl'
    print(f'=========================================================')
    print(f'Using parallel is {use_parallel}, Using raw is {use_raw}')
    # print(f'Using modeling_file is {args.modeling_file}')
    print(f'[RunID={args.run_id}] Model path is {model_path}')
    print(f'Run bench is {args.bench_name}: {bench_file}')
    print(f'Attn_implementation is {attn_implementation}')
    print(f'Modeling file is {Qwen2ForParaCausalLM}')
    print(f'=========================================================')

    eval_save_dir = os.path.join(args.base_save_dir, args.bench_name, 'model_answer')
    log_save_dir = os.path.join(args.base_save_dir, args.bench_name, 'run_details')
    os.makedirs(eval_save_dir, exist_ok=True)
    os.makedirs(log_save_dir, exist_ok=True)

    if not use_parallel:
        args.modeling_file = ''
    run_name = f'{model_name}_run{args.run_id}'
    run_name = run_name.replace('/', '_')
    model_id = run_name
    eval_save_file = os.path.join(eval_save_dir, f'{run_name}.jsonl')
    detail_save_file = os.path.join(log_save_dir, f'{run_name}.jsonl')
    detail_save_xls = os.path.join(log_save_dir, f'{run_name}.xlsx')

    input_data, save_data, next_data, para_stats_arr_all = reload_test_data(args,
                                                                            bench_file, detail_save_file,
                                                                            eval_save_file)

    if len(next_data) == 0:
        exit(0)
    model, tokenizer, normal_model, normal_tokenizer = determine_model(args, model_path, attn_implementation)

    para_st_token = 151665
    para_ed_token = 151666
    import logging

    idx = 0
    # save_data = []
    for d in tqdm(next_data):
        idx += 1
        print(f'Proccessing: {idx}', d.keys())
        print()
        normal_outputs = []
        normal_messages = [{"role": "system", "content": "You are a helpful assistant."}]
        para_outputs = []
        para_messages = [{"role": "system", "content": "You are a helpful assistant."}]
        para_stats_arr = []
        done = True
        queris = d["turns"]
        save_turns = []
        for q in queris:
            print(f'Current problem: [{q}]')
            torch.cuda.empty_cache()
            time.sleep(1)
            normal_text = ''
            para_text = ''
            normal_text = ''
            normal_num_tokens = ''
            normal_token_speed = ''
            normal_time = ''
            para_text = ''
            para_plain_text = ''
            time1 = ''
            time2 = ''
            total_para_time = ''
            para_num_tokens = ''
            para_token_speed = ''
            para_token_speed1 = ''
            para_token_speed2 = ''
            total_para_token_speed = ''
            all_text_tokens = ''
            all_branch_content_tokens = ''
            para_rate = ''
            para_efficient = ''
            num_para = ''
            para_first_token_time = ''
            normal_first_token_time = ''

            # break
            torch.cuda.empty_cache()
            if use_parallel:
                res = para_infer(para_messages, q, tokenizer, model, para_st_token, para_ed_token)
                save_turns.append(res['para_text'])
            else:
                res = normal_infer(normal_messages, q, normal_tokenizer, normal_model)
                save_turns.append(res['normal_text'])

            normal_text = res['normal_text']
            para_text = res['para_text']
            if normal_text == '' and para_text == '':
                done = False
                print(f"normal_text == '' and para_text == ''...........")
                break

            final_res = {
                'question': q,
                'category': d['category']
            }
            final_res.update(res)
            para_stats_arr.append(final_res)

        if not done:
            continue
        para_stats_arr_all += para_stats_arr[-len(queris):]
        keys = list(para_stats_arr[-1].keys())
        write2excel(detail_save_xls, para_stats_arr_all, keys)
        write2jsonline(detail_save_file, para_stats_arr_all)
        d['para_messages'] = para_messages
        d['normal_messages'] = normal_messages
        d['para_stats'] = para_stats_arr
        d['answer_id'] = shortuuid.uuid()
        d['model_id'] = model_id
        d['choices'] = [{"index": 0, "turns": save_turns}]
        d['tstamp'] = time.time()
        save_data.append(d)
        write2jsonline(eval_save_file, save_data)
    write2excel(detail_save_xls, para_stats_arr_all, list(para_stats_arr_all[-1].keys()))





