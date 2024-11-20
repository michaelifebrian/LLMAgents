import json
import os
from transformers.utils import get_json_schema
from flask import Flask, Response, stream_with_context, request, render_template, jsonify, send_from_directory
from tools import searchengine, scrapeweb, stable_diffusion_generate_image, pythoninterpreter
from utils import parseoutput, query, generate_prompt
from pyngrok import ngrok
from sseclient import SSEClient
import time

public_url = ngrok.connect(5000).public_url
print(f"Access this: {public_url}")

apimodel = "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo"
URL = "https://api.together.xyz"
getFunction = {
    "searchengine": searchengine,
    "scrapeweb": scrapeweb,
    "stable_diffusion_generate_image": stable_diffusion_generate_image,
}
TOOLS = [stable_diffusion_generate_image,
         scrapeweb,
         searchengine,
         ]
tools = [get_json_schema(i) for i in TOOLS]
tools = "\n".join([f"Use the function '{i['function']['name']}' to: '{i['function']['description']}'\n{i['function']}" for i in tools])
toolsAlias = {
    "stable_diffusion_generate_image": "Stable Diffusion Image Generator",
    "scrapeweb": "Web Scrapper",
    "searchengine": "Web Search",
}
system_prompt = f"""Environment: ipython
You are a helpful assistant. Response with a cheerful and accurate answer.

Use functions if needed and it can help you to assist users better. Think very carefully before calling functions.
You have access to following functions:

{tools}

If a you choose to call a function ONLY reply in the following format:
<{{start_tag}}={{function_name}}>{{parameters}}{{end_tag}}
where

start_tag => `<function`
parameters => a JSON dict with the function argument name as key and function argument value as value.
end_tag => `</function>`

Here is an example,
<function=example_function_name>{{"example_name": "example_value"}}</function>

Reminder:
- Function calls MUST follow the specified format, remember to start with <function= and end with </function> 
- Required parameters MUST be specified
- Only call one function at a time
- Put the entire function call reply on one line
- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls
- If the function return image filename, display the image to users with markdown format '![image-title](image-filename)'
- You may provide more information to users after getting the results from function

Also you have built in python interpreter. To run python code, just call the <|python_tag|> followed by the code. This will return the output within the image file, display it with '![image-title](image-filename)'"""
print(system_prompt)
seed = int.from_bytes(os.urandom(8), 'big')
max_tokens = 8000
chat = []
userturn = None


def resetconv():
    global userturn
    global chat
    chat = [{'role': 'system', 'content': system_prompt}]
    userturn = True


def runmodel(usertext):
    global userturn
    global chat
    if userturn:
        print("---" * 1000)
        chat.append({'role': 'user', 'content': usertext})
        print("---" * 1000)
        print("AI: \n", end="")
        userturn = False
    while not userturn:
        prompt = generate_prompt(chat) + "<|start_header_id|>assistant<|end_header_id|>\n\n"
        fulloutput = ""
        try:
            time.sleep(1)
            data = query(URL,
                         {"model": apimodel,
                          "prompt": prompt,
                          "max_tokens": 4096,
                          'stream': True,
                          })
            client = SSEClient(data)
        except Exception as e:
            yield f"error: {e}"
            break
        toolcalls = False
        for token in client.events():
            print(token.data)
            if token.data != "[DONE]":
                chunk = json.loads(token.data)['choices'][0]["text"]
                finish_reason = json.loads(token.data)['choices'][0]["finish_reason"]
                print("STREAM: " + chunk)
                fulloutput += chunk
                if chunk == "<|python_tag|>":
                    toolcalls = True
                yield chunk if not toolcalls else ''
        userturn = True if finish_reason == "eos" else False
        output = fulloutput
        print("parsing " + output)
        parsed = parseoutput(output)
        if parsed[0] is not None:
            chat.append({'role': 'assistant', 'content': parsed[2]})
            yield f"**{toolsAlias[parsed[0]]} called.**\n\n"
            if parsed[0] != 'pythoninterpreter':
                chat.append({
                    'role': 'ipython',
                    'content': getFunction[parsed[0]](**parsed[1])
                })
        elif parsed[0] == None and parsed[2] != "Error":
            chat.append({'role': 'assistant', 'content': output})
            if "<|python_tag|>" in output:
                yield "**Python Interpreter called.**\n\n"
                outputpython = pythoninterpreter(output.replace("<|python_tag|>", ""))
                yield f'```python\n{output.replace("<|python_tag|>", "")}\n```\n\n'
                chat.append({'role': 'ipython', 'content': outputpython})
        elif parsed[0] == None and parsed[2] == "Error":
            chat.append({'role': 'assistant', 'content': output})
            chat.append({'role': 'ipython', 'content': 'Error Parsing JSON'})

app = Flask(__name__)


# Update any base URLs to use the public ngrok URL
app.config["BASE_URL"] = public_url


@app.route('/sendtext')
def send_text():
    usertext = request.args.get('text')
    response = runmodel(usertext)
    return Response(stream_with_context(response))


@app.route('/<path:filename>')
def download_file(filename):
    return send_from_directory("", filename, as_attachment=False)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/resetconv', methods=['POST'])
def reset_conversation():
    resetconv()
    return jsonify({'message': 'Conversation reset'}), 200


@app.route('/chatdict')
def chat_history():
    return jsonify(chat)


if __name__ == '__main__':
    print(f"Using model: {apimodel}")
    resetconv()
    app.run(debug=False)
