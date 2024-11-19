import re
import json
import requests

HF_TOKEN = ""


def parseoutput(output):
    functions_name = []
    functions_args = []
    tool_call_match = re.search(r'<tool_call>(.*?)</tool_call>', output, re.DOTALL)
    while tool_call_match:
        tool_call_match = re.search(r'<tool_call>(.*?)</tool_call>', output, re.DOTALL)
        if tool_call_match:
            tool_call_json = tool_call_match.group(1)
            output = output.replace("\n<tool_call>" + tool_call_json + "</tool_call>", "")
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
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json",
    "x-wait-for-model": "true",
    "x-use-cache": "false"
}
def query(url, payload):
    responsetext = requests.post(url + "/v1/completions", headers=headers, json=payload)
    return responsetext.json()