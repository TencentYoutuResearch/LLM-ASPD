import logging
import requests
import os


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
