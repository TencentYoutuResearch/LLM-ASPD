# %%
#!/usr/bin/env python
#-*- coding:utf-8 -*-
import logging
import sys
import json
import os
import os.path as osp
import tqdm
os.environ['TOKENIZERS_PARALLELISM']='false'
import openai
import re
# %%
# 线程池帮助类
import concurrent.futures

def make_task(func, *args, **kwargs):
    return {
        'func': func,
        'arg': args,
        'kwargs': kwargs
    }

import concurrent.futures

class ThreadPool:
    def __init__(self, max_workers=5):
        self.max_workers = max_workers
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers)

    def submit_tasks(self, tasks):
        results = {}
        future_to_key = {}
        for key, task in tasks.items():
            func = task['func']
            args = task.get('arg', ())
            kwargs = task.get('kwargs', {})
            future = self.executor.submit(func, *args, **kwargs)
            future_to_key[future] = key

        for future in concurrent.futures.as_completed(future_to_key):
            key = future_to_key[future]
            try:
                results[key] = future.result()
            except Exception as exc:
                results[key] = exc  # 你可以选择如何处理异常
        return results

    def shutdown(self):
        self.executor.shutdown()

# %%
# Max threads
PARA_NUM = int(os.getenv('PARA_NUM', 60))
print('Start with PARA_NUM:', PARA_NUM)
API_POOL = ThreadPool(max_workers=PARA_NUM*2)


from transformers import AutoTokenizer
# Tokenizer
model_path = os.getenv('MODEL_PATH', None)
print('Using tokenizer:', model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

import sys
sys.path.append('..')
from openai_api import chat_completions
from metrics import analyze_asyncgroups_text
from misc import *
# Call OpenAI API
def chat_completions_of(messages, n=3, temperature=1.0,top_p=0.95,max_new_tokens=18024):
    return chat_completions(messages, n=n, temperature=temperature,top_p=top_p,max_new_tokens=max_new_tokens)



# %%
# 读取各种判断prompt模板
with open('prompts/para_rewrite.txt', 'r', encoding='utf-8') as fr:
    REWRITE_PROMPT = fr.read()
with open('prompts/para_indep.txt', 'r', encoding='utf-8') as fr:
    INDEPENDT_PROMPT = fr.read()
with open('prompts/para_format.txt', 'r', encoding='utf-8') as fr:
    FORMAT_PROMPT = fr.read()
with open('prompts/para_right.txt', 'r', encoding='utf-8') as fr:
    RIGHT_PROMPT = fr.read()

# 快速构建prompt模板
def make_rewrite_prompt(question_or_prompt, raw_answer):
    global REWRITE_PROMPT
    return REWRITE_PROMPT.format(answer=raw_answer)

import re
# 全局正则表达式编译
ASYNCGROUP_PATTERN = re.compile(r'<branchgroup>(.*?)</branchgroup>', re.DOTALL)
ASYNC_PATTERN = re.compile(r'<branch\s+title="(.*?)">(.*?)</branch>', re.DOTALL)

def extract_async_sync(text):
    result = []
    pos = 0
    for m in ASYNCGROUP_PATTERN.finditer(text):
        start, end = m.span()
        # 处理<branchgroup>前的sync内容
        if pos < start:
            sync_content = text[pos:start]
            if sync_content:
                result.append({"type": "sync", "content": sync_content})
        # 处理branchgroup内容
        group_content = m.group(1)
        asyncs = []
        for am in ASYNC_PATTERN.finditer(group_content):
            title = am.group(1)
            content = am.group(2)
            asyncs.append({"title": title, "content": content})
        result.append({"type": "async", "content": asyncs})
        pos = end
    # 处理最后一个</branchgroup>后的sync内容
    if pos < len(text):
        sync_content = text[pos:]
        if sync_content:
            result.append({"type": "sync", "content": sync_content})
    return result

forbiden_words = ['[PARALLEL]', '[OUTPUT_START]','[OUTPUT_END]', '【Parallel Text】']

def make_rewrite_task(question_or_prompt, raw_answer, n):
    """
    Create rewrite tasks based on the given question/prompt and raw answer.
    
    Args:
        question_or_prompt: The input question or prompt
        raw_answer: The original answer text
        n: Number of completions to generate
    
    Returns:
        List of rewrite task results
    """
    global API_POOL, tokenizer
    
    messages = [{"role": "user", "content": make_rewrite_prompt(question_or_prompt, raw_answer)}]
    task_res = chat_completions_of(messages, n=n)
    
    return [process_single_result(text, raw_answer) for text in task_res]


def process_single_result(text, raw_answer):
    """Process a single completion result."""
    is_parallel = extract_parallel_flag(text)
    para_text = None
    para_infos = []
    
    if is_parallel:
        para_text = extract_output_text(text)
        validate_text(para_text)
        para_infos = process_parallel_content(para_text)
        is_parallel = has_actual_parallelism(para_infos)
    
    # If not parallel, use the original answer
    if not is_parallel:
        para_infos = [{"type": "sync", "content": raw_answer}]
    
    return {
        "rewrite": is_parallel,
        "rewrite_text": para_text,
        "rewrite_votes": [],
        "rewrite_reasons": [],
        "para_infos": para_infos,
    }


def extract_parallel_flag(text):
    """Extract the parallel flag from the text."""
    start = text.find('[PARALLEL]')
    end = text.rfind('[PARALLEL]')
    
    if start == -1 or end == -1:
        return False
    
    flag_text = text[start + len('[PARALLEL]'):end].strip()
    return flag_text == 'true'


def extract_output_text(text):
    """Extract the output text between markers."""
    start = text.find('[OUTPUT_START]')
    end = text.rfind('[OUTPUT_END]')
    
    if start == -1 or end == -1:
        return None
    
    return text[start + len('[OUTPUT_START]'):end].strip()


def validate_text(para_text):
    """Validate that the text doesn't contain forbidden words."""
    if not para_text:
        return
    
    for fw in forbiden_words:
        if fw in para_text:
            raise RuntimeError(f'Forbidden character detected in rewritten text: {fw}')


def process_parallel_content(para_text):
    """Process parallel content and convert single-branch async to sync."""
    para_infos = extract_async_sync(para_text)
    new_para_infos = []
    
    for p in para_infos:
        if should_convert_to_sync(p):
            sync_content = merge_async_content(p["content"])
            if sync_content.strip():
                new_para_infos.append({"type": "sync", "content": sync_content})
        else:
            new_para_infos.append(p)
    
    return new_para_infos


def should_convert_to_sync(para_info):
    """Check if async block should be converted to sync (single branch or less)."""
    return para_info['type'] == "async" and len(para_info["content"]) <= 1


def merge_async_content(content_list):
    """Merge async content items into a single string."""
    return '\n'.join([item["content"] for item in content_list])


def has_actual_parallelism(para_infos):
    """Check if there's actual parallelism (at least one async block with multiple branches)."""
    return any(p['type'] == "async" and len(p.get("content", [])) > 1 for p in para_infos)


def make_independent_prompt(question_or_prompt, answer_prefix, branch_infos):
    global INDEPENDT_PROMPT
    branch_str = ''
    for i,p in enumerate(branch_infos):
        branch_str += f'<Branch{i+1} title="{p["title"]}">{p["content"]}\n</Branch{i+1}>\n'
    return (INDEPENDT_PROMPT.replace('{question}', question_or_prompt)
        .replace('{answer_prefix}', answer_prefix)
        .replace('{branch_infos}', branch_str))

def make_independent_task(question_or_prompt, para_infos, n, num_independent_accept):
    """
    Process paragraph information to determine task independence.
    
    Args:
        question_or_prompt: The question or prompt text
        para_infos: List of paragraph information dictionaries
        n: Number of completions to generate
        num_independent_accept: Minimum votes needed to accept independence
    
    Returns:
        Dictionary containing independence analysis results
    """
    global API_POOL, tokenizer
    
    answer_prefix = ''
    new_para_infos = []
    all_independent_votes = []
    all_reasons = []
    
    for para_info in para_infos:
        if para_info["type"] == "sync":
            # Process synchronous paragraph
            answer_prefix, new_para_infos = _process_sync_paragraph(
                para_info, answer_prefix, new_para_infos
            )
        else:  # async type
            # Process asynchronous paragraph
            votes, reasons, updated_para_info, answer_prefix = _process_async_paragraph(
                question_or_prompt, answer_prefix, para_info, n, num_independent_accept
            )
            all_independent_votes.append(votes)
            all_reasons.append(reasons)
            new_para_infos.append(updated_para_info)
    
    # Check if there are any async branch groups
    has_branch_groups = any(p["type"] == "async" for p in new_para_infos)
    
    return {
        "indepent": has_branch_groups,
        "indepent_votes": all_independent_votes,
        "indepent_reasons": all_reasons,
        "new_para_infos": new_para_infos,
    }


def _process_sync_paragraph(para_info, answer_prefix, new_para_infos):
    """Process synchronous paragraph information."""
    answer_prefix += para_info["content"]
    new_para_infos.append(para_info)
    return answer_prefix, new_para_infos


def _process_async_paragraph(question_or_prompt, answer_prefix, para_info, n, num_independent_accept):
    """
    Process asynchronous paragraph and determine if it should remain async.
    
    Returns:
        Tuple of (votes, reasons, updated_para_info, updated_answer_prefix)
    """
    # Generate independence analysis messages
    messages = [{
        "role": "user",
        "content": make_independent_prompt(question_or_prompt, answer_prefix, para_info["content"])
    }]
    
    # Get completions and extract votes/reasons
    task_results = chat_completions_of(messages, n=n)
    votes, reasons = _extract_votes_and_reasons(task_results)
    
    # Convert async content to sync text
    async_lines = [pi["content"] for pi in para_info["content"]]
    sync_text = '\n'.join(async_lines)
    
    # Determine if paragraph should remain async based on votes
    if sum(votes) >= num_independent_accept:
        updated_para_info = para_info  # Keep as async
    else:
        updated_para_info = {"type": "sync", "content": sync_text}  # Convert to sync
    
    answer_prefix += sync_text
    return votes, reasons, updated_para_info, answer_prefix


def _extract_votes_and_reasons(task_results):
    """
    Extract independence votes and reasons from task results.
    
    Returns:
        Tuple of (votes_list, reasons_list)
    """
    votes = []
    reasons = []
    
    for text in task_results:
        # Extract independence vote
        is_independent = _extract_between_tags(text, '[PARALLEL]', '[PARALLEL]')
        vote = 1 if is_independent.strip() == 'true' else 0
        votes.append(vote)
        
        # Extract reason
        reason = _extract_between_tags(text, '[REASON_START]', '[REASON_END]')
        reasons.append(reason.strip())
    
    return votes, reasons


def _extract_between_tags(text, start_tag, end_tag):
    """Extract text between specified tags."""
    start_index = text.find(start_tag)
    end_index = text.rfind(end_tag)
    
    if start_index == -1 or end_index == -1:
        return ""
    
    return text[start_index + len(start_tag):end_index]

def convert_to_plain_format(parsed_list):
    output = []
    for item in parsed_list:
        if item["type"] == "sync":
            output.append(item["content"])
        elif item["type"] == "async":
            async_lines = [f'{async_item["content"]}' for async_item in item["content"]]
            output.append('\n'.join(async_lines))
    return ''.join(output)

def make_format_prompt(question_or_prompt, raw_answer, para_infos):
    global FORMAT_PROMPT
    return (FORMAT_PROMPT.replace('{raw_answer}', raw_answer)
        .replace('{model_answer}', convert_to_plain_format(para_infos)))

def make_format_task(question_or_prompt, raw_answer, para_infos, n, num_format_accpet):
    global API_POOL,tokenizer
    messages = [{"role": "user", "content": make_format_prompt(question_or_prompt, raw_answer, para_infos)}]
    task_res = chat_completions_of(messages, n=n)
    format_votes = []
    reasons = []
    for text in task_res:
        f = text.find('[SAME_FOREMAT]')
        r = text.rfind('[SAME_FOREMAT]')
        is_format = text[f+len('[SAME_FOREMAT]'):r].strip()
        is_format = is_format == 'true'
        f = text.find('[REASON_START]')
        r = text.rfind('[REASON_END]')
        reason = text[f+len('[REASON_START]'):r].strip()
        format_votes.append(1 if is_format is True else 0)
        reasons.append(reason)
    is_format = sum(format_votes) >= num_format_accpet
    return {
        "format": is_format,
        "format_votes": format_votes,
        "format_reasons": reasons,
    }

def make_right_prompt(question_or_prompt, raw_answer, para_infos):
    global RIGHT_PROMPT
    return (RIGHT_PROMPT.replace('{raw_answer}', raw_answer)
        .replace('{model_answer}', convert_to_plain_format(para_infos)))

def make_right_task(question_or_prompt, raw_answer, para_infos, n, num_right_accpet):
    global API_POOL,tokenizer
    messages = [{"role": "user", "content": make_right_prompt(question_or_prompt, raw_answer, para_infos)}]
    task_res = chat_completions_of(messages, n=n)
    right_votes = []
    reasons = []
    for text in task_res:
        f = text.find('[SAME_ANSWER]')
        r = text.rfind('[SAME_ANSWER]')
        is_right = text[f+len('[SAME_ANSWER]'):r].strip()
        is_right = is_right == 'true'
        f = text.find('[REASON_START]')
        r = text.rfind('[REASON_END]')
        reason = text[f+len('[REASON_START]'):r].strip()
        right_votes.append(1 if is_right is True else 0)
        reasons.append(reason)
    is_right = sum(right_votes) >= num_right_accpet
    return {
        "right": is_right,
        "right_votes": right_votes,
        "right_reasons": reasons,
    }

setup_seed(1024)

def adapt_old_data(para_stas):
    para_stas['平均分支数量'] = para_stas['abn']
    para_stas['平均并行度'] = para_stas['apd']
    para_stas['平均并行效率'] = para_stas['ape']
    return para_stas

def make_para_data(question_or_prompt, raw_answer,min_skip_tokens=128,sample_size=3,
        num_rewrite_accept=3, num_independent_accept=3,num_format_accpet=3,num_right_accpet=3
    ):
    """
    对单条数据进行清洗和处理，生成用于并行解码的训练样本。

    参数:
        question_or_prompt (str): 输入的问题或提示语。
        raw_answer (str): 原始答案文本。
        min_skip_tokens (int, 可选): 跳过清洗的最小token数，默认128。
        sample_size (int, 可选): 每个清洗阶段从采样量，默认3，这用于后面每个阶段的投票基数。
        num_rewrite_accept (int, 可选): 可接受的重写样本阈值，默认3。
        num_independent_accept (int, 可选): 可接受的独立样本阈值，默认3。
        num_format_accpet (int, 可选): 可接受的格式正确样本阈值，默认3。
        num_right_accpet (int, 可选): 可接受的正确样本阈值，默认3。

    返回:
        dict: 包含清洗和处理后数据的字典，具体结构根据实现而定。

    示例:
        data = make_para_data("什么是机器学习？", "机器学习是一种人工智能方法...")
    """
    # 函数实现
    global THREAD_POOL,tokenizer
    result = []
    if len(tokenizer.encode(raw_answer))<min_skip_tokens:
        result.append([{
            "para_infos": None,
            "rewrite": None,
            "rewrite_text": None,
            "rewrite_votes": [],
            "rewrite_reasons": [],
            "indepent": None,
            "indepent_votes": [],
            "indepent_reasons": [],
            "format": None,
            "format_votes": [],
            "format_reasons": [],
            "right": None,
            "right_votes": [],
            "right_reasons": [],
        }, raw_answer])
        return result
    rewrite_res = make_rewrite_task(question_or_prompt, raw_answer, n=sample_size)
    for r in rewrite_res:
        para_stas = adapt_old_data(analyze_asyncgroups_text([r["para_infos"]], tokenizer))
        r["new_para_infos_stas"] = para_stas
        if r["rewrite"]:
            indepent_res = make_independent_task(question_or_prompt, r["para_infos"], 
                n=sample_size, num_independent_accept=num_independent_accept)
            r.update(indepent_res)
            if r["indepent"]:
                new_para_infos = r["new_para_infos"]
                para_stas = adapt_old_data(analyze_asyncgroups_text([new_para_infos], tokenizer))
                r["new_para_infos_stas"] = para_stas
                format_task = make_task(make_format_task, question_or_prompt, raw_answer, 
                    new_para_infos, n=sample_size, num_format_accpet=num_format_accpet)
                right_task = make_task(make_right_task, question_or_prompt, raw_answer, 
                    new_para_infos, n=sample_size, num_right_accpet=num_right_accpet)
                task_res = API_POOL.submit_tasks({
                    "format_task": format_task,
                    "right_task": right_task,
                })
                # print(task_res)
                for t, res in task_res.items():
                    r.update(res)
                if r["right"] and r["format"]:
                    result.append([r, new_para_infos])
                    continue
            else:
                r.update({
                    "format": None,
                    "format_votes": [],
                    "format_reasons": [],
                    "right": None,
                    "right_votes": [],
                    "right_reasons": [],
                })
        else:
            # 设置并行独立性判断、格式一致性和答案一致性判断信息为空
            r.update({
                "indepent": None,
                # "indepdent_text": None,
                "indepent_votes": [],
                "indepent_reasons": [],
                "format": None,
                "format_votes": [],
                "format_reasons": [],
                "right": None,
                "right_votes": [],
                "right_reasons": [],
            })
        result.append([r, raw_answer])
    return result

# %%
import logging
def make_ai_data(case_data):
    try:
        prompt = case_data['problem']
        raw_answer = case_data['raw_answer']
        return case_data, make_para_data(prompt, raw_answer)
    except Exception as e:
        logging.exception(e)
        return case_data, None

save_dir = 'para_ppl_qwen3'
os.makedirs(save_dir, exist_ok=True)

def launch_parallel_tasks(next_data, file_name):
    global PARA_NUM
    if next_data:
        import multiprocessing
        # 初始化进程池
        with multiprocessing.Pool(processes=PARA_NUM) as pool:
            # 并发执行任务并异步获取结果
            results = pool.imap_unordered(make_ai_data, next_data)
            # 处理结果并写入文件
            with open(file_name, 'a', encoding='utf-8') as fw:
                for problem_inst, result in tqdm.tqdm(results, total=len(next_data)):
                    if result:
                        problem_inst["raw_results"] = result
                        fw.write(json.dumps(problem_inst, ensure_ascii=False, indent=None) + '\n')

def filter_data(input_data):
    short_data = []
    new_input_data = []
    # 过滤超出训练长度的数据
    for d in tqdm.tqdm(input_data):
        raw_answer = None
        if 'target' in d:
            problem = d['input']
            raw_answer = d['target']
            history = []
        elif 'conversations' in d:
            messages = d['conversations']
            assert len(messages)%2 == 0
            message_str = ''
            history = []
            for i in range((len(messages)//2) - 1):
                message_str = f"{messages[2*i]['from']}: {messages[2*i]['value']}\n"
                message_str = f"{messages[2*i+1]['from']}: {messages[2*i+1]['value']}\n"
            problem = message_str + f"{messages[-2]['from']}: {messages[-2]['value']}\n"
            raw_answer = messages[-1]['value']
        if len(tokenizer.encode(problem+raw_answer))>8000:
            continue
        d['raw_answer'] = raw_answer
        d['problem'] = problem
        if len(tokenizer.encode(raw_answer))<128:
            short_data.append(d)
            continue
        new_input_data.append(d)
    return short_data,new_input_data

def get_next_data(save_data, input_data, QUESTION_KEY):
    if len(save_data)<len(input_data):
        map_data = {}
        for d in save_data:
            qkey = d[QUESTION_KEY]
            map_data[qkey] = d
        next_data = []
        for d in input_data:
            qkey = d[QUESTION_KEY]
            if qkey not in map_data:
                next_data.append(d)
    else:
        next_data = []
    return next_data

def main(path):
    print(f'path={path}')
    run_id = '_run0'
    if len(sys.argv)>2:
        run_id = '_run'+sys.argv[2]
        run_id = '_run'+'0'
    save_data = []
    file_name = os.path.join(save_dir,osp.splitext(os.path.basename(path))[0]) + f'_para{run_id}.jsonl'
    short_file_name = os.path.join(save_dir,osp.splitext(os.path.basename(path))[0]) + f'_para{run_id}_short.jsonl'
        
    save_data = []
    if os.path.isfile(file_name):
        save_data = read2jsonline(file_name)

    QUESTION_KEY = 'index'
    if 'unstructured' in path:
        QUESTION_KEY = 'id'
    input_data = read2jsonline(path)
    short_data,new_input_data = filter_data(input_data)
    
    write2jsonline(short_file_name, short_data)
    del short_data
    input_data = new_input_data
    print(f'reload {file_name} from {len(save_data)}, total {len(input_data)}')
    next_data = get_next_data(save_data, input_data, QUESTION_KEY)
    launch_parallel_tasks(next_data, file_name)

if __name__ == '__main__':
    main(sys.argv[1])


