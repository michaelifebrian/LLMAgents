import requests
from apitoken import API_TOKEN

headers = {
    "Authorization": f"Bearer {API_TOKEN}",
    "Content-Type": "application/json",
    "Accept": "application/json"
}


def query(url, payload):
    return requests.post(url + "/v1/completions", headers=headers, json=payload, stream=True)


def generate_prompt(chat_dict):
    prompt = "<|begin_of_text|>"
    for i in chat_dict:
        prompt += f"<|start_header_id|>{i['role']}<|end_header_id|>\n\n{i['content']}"
    return prompt
