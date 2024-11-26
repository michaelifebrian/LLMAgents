import requests
from apitoken import API_TOKEN
import json
from transformers.utils import get_json_schema

headers = {
    "Authorization": f"Bearer {API_TOKEN}",
    "Content-Type": "application/json",
    "Accept": "application/json"
}


def create_tools_json(tools):
    getFunction = {}
    for i in tools:
        getFunction[get_json_schema(i)['function']['name']] = i
    tools = [json.dumps(get_json_schema(i), indent=4) for i in tools]
    tools = "\n".join(tools)
    return getFunction, tools


def is_json(input_json):
    try:
        json.loads(input_json)
    except ValueError:
        return False
    return True


def query(url, payload, prompt):
    payload['prompt'] = prompt
    return requests.post(url + "/v1/completions", headers=headers, json=payload, stream=True)


def generate_prompt(chat_dict):
    prompt = "<|begin_of_text|>"
    for i in chat_dict:
        prompt += f"<|start_header_id|>{i['role']}<|end_header_id|>\n\n{i['content']}"
    return prompt
