import json
import os
from transformers.utils import get_json_schema
from flask import Flask, Response, stream_with_context, request, render_template, jsonify, send_from_directory
from tools import searchengine, browser, flux_generate_image, python_interpreter, is_json
from utils import query, generate_prompt
# from pyngrok import ngrok
from sseclient import SSEClient
import time
from apitoken import URL, apimodel

# public_url = ngrok.connect(5000).public_url

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

Also you have built in python interpreter. To run python code, just call the <|python_tag|> followed by the code. This will run the code in a stateful Jupyter notebook environment and automatically return the snapshot of the cell output to users.

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
userTurn = None


def reset_conv():
    global userTurn
    global chat
    chat = [{'role': 'system', 'content': system_prompt}]
    userTurn = True


def run_model(user_text):
    global userTurn
    global chat
    if userTurn:
        print("---" * 1000)
        chat.append({'role': 'user', 'content': user_text + "<|eot_id|>"})
        print("---" * 1000)
        print("AI: \n", end="")
        userTurn = False
    while not userTurn:
        prompt = generate_prompt(chat) + "<|start_header_id|>assistant<|end_header_id|>\n\n"
        output = ""
        print(prompt)
        try:
            # time.sleep(1) #To limit RPM
            data = query(URL,
                         {"model": apimodel,
                          "prompt": prompt,
                          "max_tokens": 4096,
                          'stream': True,
                          'stop': ["<|eom_id|>"],
                          })
            client = SSEClient(data)
        except Exception as e:
            yield f"error: {e}"
            break
        toolCalls = False
        pythonCode = False
        finishReason = None  # None means <|eot_id|>, 128008 means <|eom_id|>
        for token in client.events():
            print(token.data)
            if token.data != "[DONE]":
                chunk = json.loads(token.data)['choices'][0]["text"]
                finishReason = json.loads(token.data)['choices'][0]["stop_reason"]
                output += chunk
                if "<|p" in chunk:
                    toolCalls = True
                if "<|python_tag|>" in output and toolCalls:
                    try:
                        nextChar = output.split("<|python_tag|>")[-1][0]
                    except:
                        nextChar = ''
                    if nextChar != "{" and not pythonCode:
                        yield "**Running python code**\n\n"
                        pythonCode = True
                        cleaned = False
                        yield '```python\n'
                if not toolCalls:
                    yield chunk
                if pythonCode:
                    if not cleaned:
                        if chunk != chunk.split(">")[-1]:
                            chunk = chunk.split(">")[-1]
                            cleaned = True
                    yield chunk
        yield '\n```\n' if pythonCode else '\n\n'
        print("parsing " + output)
        if output != "":
            userTurn = False if finishReason == 128008 else True
            suffix = "<|eom_id|>" if finishReason == 128008 else "<|eot_id|>"
            if not toolCalls:
                chat.append({'role': 'assistant', 'content': output+suffix})
            elif toolCalls:
                if is_json(output.replace("<|python_tag|>", "")):
                    parsed = json.loads(output.replace("<|python_tag|>", ""))
                    chat.append({'role': 'assistant', 'content': output+suffix})
                    yield f"\n\n**{toolsAlias[parsed['name']]} called.**\n\n"
                    chat.append({
                        'role': 'ipython',
                        'content': json.dumps(getFunction[parsed['name']](**parsed['parameters'])) + "<|eot_id|>"
                    })
                else:
                    chat.append({'role': 'assistant', 'content': output+suffix})
                    outputPython = python_interpreter(output.replace("<|python_tag|>", ""))
                    chat.append({'role': 'ipython', 'content': json.dumps(outputPython) + "<|eot_id|>"})
                    if outputPython['cell_snapshot'] != 'Error running code':
                        yield f"![output]({outputPython['cell_snapshot']})"
                        chat.append({'role': 'assistant', 'content': f"![output]({outputPython['cell_snapshot']})" + "<|eom_id|>"})
        else:
            userTurn = False


app = Flask(__name__)


# app.config["BASE_URL"] = public_url


@app.route('/sendtext')
def send_text():
    userText = request.args.get('text')
    response = run_model(userText)
    return Response(stream_with_context(response))


@app.route('/<path:filename>')
def download_file(filename):
    return send_from_directory("", filename, as_attachment=False)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/resetconv', methods=['POST'])
def reset_conversation():
    reset_conv()
    return jsonify({'message': 'Conversation reset'}), 200


@app.route('/chatdict')
def chat_history():
    return jsonify(chat)


if __name__ == '__main__':
    print(f"Using model: {apimodel}")
    # print(f"Access this: {public_url}")
    reset_conv()
    app.run(debug=False)
