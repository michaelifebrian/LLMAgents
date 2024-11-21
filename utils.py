import re
import json
import requests

API_TOKEN = ""

def parseoutput(response: str):
    function_regex = r"<function=(\w+)>(.*?)</function>"
    match = re.search(function_regex, response)
    if not match:
        function_regex = r"<function=([a-zA-Z_][a-zA-Z0-9_]*)>\s*(\{.*\})"
        match = re.search(function_regex, response)
    if match:
        function_name, args_string = match.groups()
        try:
            args = json.loads(args_string)
            return function_name, args, response
        except json.JSONDecodeError as error:
            print(f"Error parsing function arguments: {error}")
            return None, None, "Error"
    return None, None, response


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
