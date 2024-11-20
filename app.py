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
<|eot_id|><|start_header_id|>user<|end_header_id|>

Use functions if needed and can help you to assist users better. Think very carefully before calling functions.
You have access to several functions. Here is a list of functions in JSON format:
{toolslist}

If you choose to call a function ONLY reply in the following format with no prefix or suffix:

<tool_call>{{\"name\": \"example_function_name\", \"arguments\": {{\"example_parameters_name\": \"example_parameters_value\"}}}}</tool_call>

Reminder:
- Function calls MUST follow the specified format, remember to start with <tool_call> and end with </tool_call>. 
- Required parameters MUST be specified
- Only call one function at a time
- Put the entire function call reply on one line
- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls
- If the function return image filename, display the image to users with markdown format '![image-title](image-filename)'
- You may provide more information after getting the results from function"""
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

        # toolcallstate = False
        for token in client.events():
            if token.data != "[DONE]":
                chunk = json.loads(token.data)['choices'][0]["text"]
                print("STREAM: " + chunk)
                # if chunk in ['<', '<tool', '<tool_call', '<tool_call>']:
                #     toolcallstate = True
                # if chunk in ['</', '</tool', '</tool_call', '</tool_call>']:
                #     toolcallstate = False
                fulloutput += chunk
                # if not toolcallstate:
                yield chunk
        output = fulloutput
        print("\n", end="")
        yield "\n"
        parsed = parseoutput(output)
        if parsed[0] != []:
            userturn = False
            chat.append({'role': 'assistant', 'content': output})
            yield "\n"
            for i in range(len(parsed[0])):
                yield f"**{toolsAlias[parsed[0][i]]} called.**"
                yield "\n\n"
                chat.append({
                    'role': 'ipython',
                    'content': getFunction[parsed[0][i]](**parsed[1][i])
                })
        elif parsed[0] == [] and parsed[2] != "ERROR":
            userturn = True
            chat.append({'role': 'assistant', 'content': output})
        elif parsed[0] == [] and parsed[2] == "ERROR":
            userturn = False
            chat.append({'role': 'ipython', 'content': 'ERROR PARSING JSON'})
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
    resetconv()
    app.run(debug=True)
