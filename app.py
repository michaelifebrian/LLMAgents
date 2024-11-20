import json
import os
from transformers.utils import get_json_schema
from flask import Flask, Response, stream_with_context, request, render_template, jsonify, send_from_directory
from tools import searchengine, scrapeweb, stable_diffusion_generate_image, pythoninterpreter
from utils import parseoutput, query, generate_prompt
from pyngrok import ngrok
from sseclient import SSEClient

public_url = ngrok.connect(5000).public_url
print(f"Access this: {public_url}")

apimodel = "accounts/fireworks/models/llama-v3p1-405b-instruct"
URL = "https://api.fireworks.ai/inference"
getFunction = {
    "searchengine": searchengine,
    "scrapeweb": scrapeweb,
    "stable_diffusion_generate_image": stable_diffusion_generate_image,
    "pythoninterpreter": pythoninterpreter,
}
TOOLS = [stable_diffusion_generate_image,
         scrapeweb,
         searchengine,
         pythoninterpreter,
         ]
tools = [get_json_schema(i) for i in TOOLS]
toolsAlias = {
    "stable_diffusion_generate_image": "Stable Diffusion Image Generator",
    "scrapeweb": "Web Scrapper",
    "searchengine": "Web Search",
    "pythoninterpreter": "Python Interpreter"
}
toolslist = "\n".join([json.dumps(tool, indent=2) for tool in tools])
system_prompt = f"""You are a helpful assistant. Response with a markdown format.

Use functions if needed and can help you to assist users better. Think very carefully before calling functions.
You have access to several functions. Here is a list of functions in JSON format:
{toolslist}

If you choose to call a function ONLY reply in the following JSON format with no prefix or suffix:

<function=example_function_name>{{\"example_name\": \"example_value\"}}</function>

Another example:

<function=example_function_name>{{\"example_parameters_name\": \"example_parameters_value\"}}</function>

Reminder:
- Function calls MUST follow the specified format, remember to start with <function= and end with </function> 
- Required parameters MUST be specified
- Only call one function at a time
- Put the entire function call reply on one line
- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls
- If the function return image filename, display the image to users with markdown format '![image-title](image-filename)'
- You may provide more information to users after getting the results from function"""
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

        for token in client.events():
            print(token.data)
            if token.data != "[DONE]":
                chunk = json.loads(token.data)['choices'][0]["text"]
                print("STREAM: " + chunk)
                fulloutput += chunk
                yield chunk
        output = fulloutput
        print("\n", end="")
        yield "\n"
        print("parsing" + output)
        parsed = parseoutput(output)
        if parsed[0] is not None:
            userturn = False
            chat.append({'role': 'assistant', 'content': parsed[2]})
            yield "\n"
            yield f"**{toolsAlias[parsed[0]]} called.**"
            yield "\n\n"
            if parsed[0] != 'pythoninterpreter':
                chat.append({
                    'role': 'ipython',
                    'content': getFunction[parsed[0]](**parsed[1])
                })
            elif parsed[0] == 'pythoninterpreter':
                chat.append({
                    'role': 'ipython',
                    'content': getFunction[parsed[0]](parsed[1])
                })
        elif parsed[0] == None and parsed[2] != "Error":
            userturn = True
            chat.append({'role': 'assistant', 'content': output})
        elif parsed[0] == None and parsed[2] == "Error":
            userturn = False
            chat.append({'role': 'assistant', 'content': output})
            chat.append({'role': 'ipython', 'content': 'Error Parsing JSON'})
        if '<|STOP|>' in output:
            return


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
