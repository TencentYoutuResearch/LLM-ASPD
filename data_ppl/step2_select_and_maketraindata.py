# %%
#!/usr/bin/env python
#-*- coding:utf-8 -*-
import sys
import os
os.environ['TOKENIZERS_PARALLELISM']='false'

from transformers import AutoTokenizer
# Tokenizer
model_path = os.getenv('MODEL_PATH', None)
print('Using tokenizer:', model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
import sys
sys.path.append('..')
from misc import *

from functools import cmp_to_key

def sort_dicts_with_indices(data, cmp):
    # 记录原始索引
    indexed_data = list(enumerate(data))
    # 排序，key为自定义比较器
    indexed_data.sort(key=cmp_to_key(lambda a, b: cmp(a[1], b[1])))
    # 拆分排序后的数据和原始索引
    sorted_data = [item[1] for item in indexed_data]
    sorted_indices = [item[0] for item in indexed_data]
    return sorted_data, sorted_indices

def make_cmp(keys, reverses=None):
    """
    keys: 属性名列表，如 ['age', 'name']
    reverses: 是否降序的布尔列表，如 [False, True]
    """
    if reverses is None:
        reverses = [False] * len(keys)
    def cmp(a, b):
        for key, rev in zip(keys, reverses):
            va, vb = a[key], b[key]
            if va != vb:
                if rev:
                    return (vb > va) - (vb < va)
                else:
                    return (va > vb) - (va < vb)
        return 0
    return cmp

# 用法示例：先按 age 升序，再按 name 降序
cmp_func = make_cmp(['平均并行度', '平均分支数量', '平均并行效率'], [True, True, True])

forbidden_words = ['<|branch|>', '<|/branch|>', '<|title|>', '<|/title|>', '<|para|>', '<|/para|>', '<|im_end|>']

def convert_to_custom_format(parsed_list):
    output = []
    for item in parsed_list:
        if item["type"] == "sync":
            output.append(item["content"])
        elif item["type"] == "async":
            # 输出所有promise标签
            promises = ''.join([f'<|title|>{async_item["title"]}<|/title|>' for async_item in item["content"]])
            output.append(promises + '<|para|>')
            # 输出每个async内容，用换行符拼接，标签为<|branch|>...</|branch|>
            for it in item['content']:
                for fw in forbidden_words:
                    assert fw not in it['content'], f'发现未清理干净的特殊token({fw})：【{parsed_list}】'
            async_lines = [f'<|branch|>{async_item["title"]}: {async_item["content"]}<|/branch|>' 
                for async_item in item["content"]]
            output.append(''.join(async_lines))
            output.append('<|/para|>')
    return ''.join(output)

def raw_asnwer2output(data):
    for d in data:
        raw_answer = d['raw_answer']
        if '</think>' in raw_answer:
            raw_answer = raw_answer.split('</think>')[-1].strip()
        d['output'] = raw_answer
    return data


if __name__ == '__main__':
    save_suffix = '_apar'
    llama_data_dir = r'train_data'
    rewrite_data_dir = r'para_ppl_qwen3'
    os.makedirs(llama_data_dir, exist_ok=True)

    target_files = [
        r'apar_flatten_Qwen3_para_run0.jsonl',r'apar_flatten_Qwen3_para_run0_short.jsonl',
        r'unstructured_Qwen3_para_run0.jsonl', r'unstructured_Qwen3_para_run0_short.jsonl',
    ]
    target_files = [os.path.join(rewrite_data_dir, f) for f in target_files]

    savename = os.path.join(llama_data_dir, f'ParaTrain{save_suffix}_qw.jsonl')
    savename_raw = os.path.join(llama_data_dir, f'SFTTrain{save_suffix}_qw.jsonl')
    train_data =[]
    for tf in target_files:
        print(f'[Current file: {tf}]')
        unique_map = set()
        unique_id = 'index'
        basename = os.path.basename(tf)
        if 'unstructured' in basename:
            unique_id = 'id'
        data = read2jsonline(tf)
        # unparallel_data =[]
        for d in data:
            if d[unique_id] in unique_map:
                print(f'Skip unique_id={unique_id} in {basename}')
                continue
            unique_map.add(d[unique_id])
            message_str = ''
            if 'unstructured' in basename:
                messages = d['conversations']
                problem = messages[-2]['value']
                raw_answer = messages[-1]['value']
                assert len(messages)%2 == 0
                history = []
                for i in range((len(messages)//2) - 1):
                    history.append([messages[2*i]['value'], messages[2*i+1]['value']])
                    message_str+= messages[2*i]['value']+messages[2*i+1]['value']
            else:
                problem = d['input']
                raw_answer = d['target']
                history = []
            if 'raw_results' in d:
                raw_results = d['raw_results']
                tmp_res = []
                for r in raw_results:
                    r = r[0]
                    if r["rewrite"] == True and r["indepent"] == True and r["format"] == True and r["right"] == True:
                        tmp_res.append(r)
                final_text = None
                para_info = {}
                if tmp_res:
                    if len(tmp_res)>1:
                        para_info_arr = [r['new_para_infos_stas'] for r in tmp_res]
                        sortdata, ori_idxs = sort_dicts_with_indices(para_info_arr,cmp_func)
                        para_stats = sortdata[0]
                        # print(f'Select para: {para_stats} in [{para_info_arr}]')
                        para_info = tmp_res[ori_idxs[0]]['new_para_infos']
                        final_text = convert_to_custom_format(para_info)
                        # break
                    else:
                        para_info = tmp_res[0]['new_para_infos']
                        final_text = convert_to_custom_format(para_info)
                
                if final_text is None:
                    final_text = raw_answer
                    para_info = [{"type":"sync", "content": raw_answer}]
                encode_ids = tokenizer.encode(problem+message_str+'  '+final_text)
                if len(encode_ids)>8000:
                    final_text = raw_answer
                    para_info = [{"type":"sync", "content": raw_answer}]
            else:
                final_text = raw_answer
            train_data.append({
                "unique_id": d[unique_id],
                "system": '',
                "instruction":problem,
                "input": '',
                "history":history,
                "raw_answer": raw_answer,
                "output": final_text,
                # "para_info":para_info
            })

    write2json(savename, train_data)
    write2json(savename_raw, raw_asnwer2output(train_data))




