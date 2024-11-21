import json
import os
from transformers.utils import get_json_schema
from flask import Flask, Response, stream_with_context, request, render_template, jsonify, send_from_directory
from tools import searchengine, browser, flux_generate_image, pythoninterpreter, isjson
from utils import query, generate_prompt
from pyngrok import ngrok
from sseclient import SSEClient
import time
from apitoken import URL, apimodel

public_url = ngrok.connect(5000).public_url

getFunction = {
    "searchengine": searchengine,
    "browser": browser,
    "flux_generate_image": flux_generate_image,
}
TOOLS = [
    flux_generate_image,
    browser,
    searchengine,
]
toolsAlias = {
    "flux_generate_image": "Flux Image Generator",
    "browser": "Browser",
    "searchengine": "Web Search",
}
tools = [json.dumps(get_json_schema(i), indent=4) for i in TOOLS]
tools = "\n".join(tools)
system_prompt = f"""Environment: ipython

Cutting Knowledge Date: December 2023
Today Date: 21 September 2024

You are Llama 3.1 trained by Meta. You are a helpful assistant. Response with a cheerful and accurate answer.
<|eot_id|><|start_header_id|>user<|end_header_id|>

Use functions if needed and can help you to assist users better. Think very carefully before calling functions.
Here is a list of functions in JSON format:

{tools}

Return function calls in JSON format.

You have ability to show an image by using the markdown format: ![<image-title>](<image-filename>)

Also you have built in python interpreter. To run python code, just call the <|python_tag|> followed by the code. This will run the code in a stateful Jupyter notebook environment and return the snapshot of the cell output to users.

Policy for each functions you need to follow:
'flux_generate_image':
1. The prompt must be in English. Translate to English if needed.
2. DO NOT ask for permission to generate the image, just do it!
3. The generated prompt sent to flux should and must be very detailed, and around 100 words long and above.

'browser':
1. You can open a url directly if one is provided by the user or you know the exact url you need to open. You can open the urls returned by the searchengine function or found on webpages.
2. Some of webs can easily detecting if you (web client) are not human. If so, tell users the web cannot be opened because of security reason.
3. Some of webs needs to authenticate or login with an account. If so, tell users the web cannot be opened because it needs to login to see.

'searchengine':
1. You can use `searchengine` in the following circumstances and not limited by:
    - User is asking about current events or something that requires real-time information (weather, sports scores, etc.)
    - User is asking about some term you are totally unfamiliar with (it might be new)
    - User explicitly asks you to search
2.Given a query that requires retrieval, your turn will consist of three steps:
    - Call the searchengine function to get a list of results.
    - If the returned results doesn't give much information, you can open the webpages by calling 'browser' function using their href urls.
    - Write a response to the user based on these results. If possible, in your response cite the sources you are referring.

Python interpreter:
When you send a message containing Python code using <|python_tag|>, it will be executed in a stateful Jupyter notebook environment. Python will respond with the output of the cells in text format and the snapshot of the cells output from the execution. 
If you do not want to run the Python code (or any code) in case you want to give the code only, use these markdown format:
```
<code>
```

Always give users some explanation of what you just do or what the results is after the function has returned a results."""
# system_prompt = "You are a helpful assistant."
system_prompt += "<|eot_id|>"
print(system_prompt)
seed = int.from_bytes(os.urandom(8), 'big')
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
        chat.append({'role': 'user', 'content': usertext + "<|eot_id|>"})
        print("---" * 1000)
        print("AI: \n", end="")
        userturn = False
    while not userturn:
        prompt = generate_prompt(chat) + "<|start_header_id|>assistant<|end_header_id|>\n\n"
        print(prompt)
        output = ""
        try:
            time.sleep(1) #To limit RPM
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
                output += chunk
                if "<|python_tag|>" in chunk:
                    toolcalls = True
                yield chunk if not toolcalls else ''
        print("parsing " + output)

        if output != "":
            if "<|python_tag|>" not in output:
                chat.append({'role': 'assistant', 'content': output + "<|eot_id|>"})
                userturn = True
            elif "<|python_tag|>" in output:
                userturn = False
                if isjson(output.replace("<|python_tag|>", "")):
                    parsed = json.loads(output.replace("<|python_tag|>", ""))
                    chat.append({'role': 'assistant', 'content': output + "<|eom_id|>"})
                    yield f"**{toolsAlias[parsed['name']]} called.**\n\n"
                    chat.append({
                        'role': 'ipython',
                        'content': json.dumps(getFunction[parsed['name']](**parsed['parameters'])) + "<|eot_id|>"
                    })
                else:
                    chat.append({'role': 'assistant', 'content': output + "<|eom_id|>"})
                    yield "**Python Interpreter called.**\n\n"
                    outputpython = pythoninterpreter(output.replace("<|python_tag|>", ""))
                    yield f'```python\n{output.replace("<|python_tag|>", "")}\n```\n\n'
                    yield f"![output]({outputpython['cell_snapshot']})" if outputpython['cell_snapshot'] != '' else ''
                    chat.append({'role': 'ipython', 'content': json.dumps(outputpython) + "<|eot_id|>"})
        else:
            userturn = False


app = Flask(__name__)


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
    print(f"Access this: {public_url}")
    resetconv()
    app.run(debug=False)