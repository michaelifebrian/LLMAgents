import re
import json
import requests

API_TOKEN = ""

def parseoutput(output):
    functions_name = []
    functions_args = []
    tool_call_match = re.search(r'<tool_call>(.*?)</tool_call>', output, re.DOTALL)
    while tool_call_match:
        tool_call_match = re.search(r'<tool_call>(.*?)</tool_call>', output, re.DOTALL)
        if tool_call_match:
            tool_call_json = tool_call_match.group(1)
            output = output.replace("<tool_call>" + tool_call_json + "</tool_call>", "")
            try:
                tool_call_data = json.loads(tool_call_json)

                # Extracting the function name and arguments
                function_name = tool_call_data.get("name")
                function_name = tool_call_data.get("function") if function_name is None else function_name
                arguments = tool_call_data.get("arguments")
                if type(arguments) == str:
                    arguments = json.loads(arguments)
                functions_name.append(function_name)
                functions_args.append(arguments)

            except json.JSONDecodeError:
                print("Error parsing JSON content.")
                output = "ERROR"
    return functions_name, functions_args, output


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
        prompt += f"<|start_header_id|>{i['role']}<|end_header_id|>\n\n{i['content']}<|eot_id|>"
    return prompt
