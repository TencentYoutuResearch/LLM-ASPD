# Non-invasive Parallel Data Transformation Pipeline



# Step 0. Add Special Tokens
Replace the path with your local model path to [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct):
```bash
export RAW_MODEL_PATH=path/to/Qwen2.5-7B-Instruct
```

Then add special tokens to tokenizer and resize token embeddings of the model:
```bash
python step0_add_sptokens.py
```

The tokenizer and model will be saved in `para_model/Qwen2.5-7B-Instruct-Async`

# Step 1. Run Data PPL

Before start, please download data files (from [APAR](https://github.com/THUDM/APAR)) into `data` dir:
```bash
wget 'https://github.com/THUDM/APAR/raw/refs/heads/main/data/unstructured.json?download=' -O data/unstructured.jsonl
wget 'https://github.com/THUDM/APAR/raw/refs/heads/main/data/apar_flatten.json?download=' -O data/apar_flatten.jsonl
````
For each data file, please rewrite model output with `Qwen3-235B-A22B` to produce `apar_flatten_Qwen3.jsonl` and `unstructured_Qwen3.jsonl` in `data` dir.

You can replace the following code with your implementation for calling qwen3 in `openai_api.py` without changing the function name `chat_completions` and corresponding signature:
```python
base_url = os.getenv('OPENAI_API_CHAT_ENDPOINT', None)

# Call OpenAI API
def chat_completions(messages, n=3, temperature=1.0,top_p=0.95,max_new_tokens=14024):
    global base_url
    fresponse = None
    try:
        post_data = {
            "model": "Qwen3-235B-A22B",
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_new_tokens,
            "n": n,
            "extra_body": {
                "chat_template_kwargs": {"enable_thinking": True},
                # "chat_template_kwargs": {"enable_thinking": False},
                "separate_reasoning": True
            }
        }
        response = requests.post(base_url, json=post_data)
        # print(response)
        response = response.json()
        # print(response)
        answer_list = []
        for choice_data in response["choices"]:
            answer = choice_data["message"]["content"]
            if '</think>' in answer:
                answer = answer.split('</think>')[-1]
            answer_list.append(answer)
        fresponse = answer_list
    except Exception as e:
        logging.exception(e)
    return fresponse
```
Or just set the environment variable of OPENAI_API_CHAT_ENDPOINT (,and you may need to change "model" value of `post_data` in the above code):
 ```bash
 # Replace the url with yours
export OPENAI_API_CHAT_ENDPOINT=https://your-OpenAI-API-Endpoint.com/v1/chat/completions
```

After that, set running args for the PPL：
```bash
export PARA_NUM=60 # Max threads for processing data
export MODEL_PATH=para_model/Qwen2.5-7B-Instruct-Async
```

Use the PPL to rewrite data and create train data:
```bash
# para data ppl
python step1_rewrite_and_check.py data/apar_flatten_Qwen3.jsonl
python step1_rewrite_and_check.py data/unstructured_Qwen3.jsonl
# create train data
python step2_select_and_maketraindata.py
```

The train data be saved in `train_data`. 
 - `ParaTrain_apar.jsonl` is for ASPD model
 - `SFTTrain_apar.jsonl` is for Seq(SFT) model