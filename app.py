import json
import os
from flask import Flask, Response, stream_with_context, request, render_template, jsonify, send_from_directory
from tools import python_interpreter
from utils import query, generate_prompt, is_json
from prompt import system_prompt, toolsAlias, getFunction, parameters
# from pyngrok import ngrok
from sseclient import SSEClient
from apitoken import URL, apimodel
# public_url = ngrok.connect(5000).public_url

seed = int.from_bytes(os.urandom(8), 'big')
chat = []
userTurn = None
stopGenerate = False


def reset_conv():
    global userTurn
    global chat
    chat = [{'role': 'system', 'content': system_prompt}]
    userTurn = True


def run_model(user_text):
    global userTurn
    global chat
    global stopGenerate
    if userTurn:
        print("---" * 1000)
        chat.append({'role': 'user', 'content': user_text + "<|eot_id|>"})
        print("---" * 1000)
        print("AI: \n", end="")
        userTurn = False
        stopGenerate = False
    while not userTurn:
        prompt = generate_prompt(chat) + "<|start_header_id|>assistant<|end_header_id|>\n\n"
        output = ""
        print(prompt)
        try:
            parameters['stream'] = True
            data = query(URL, parameters, prompt)
            client = SSEClient(data)
        except Exception as e:
            yield f"error: {e}"
            break
        toolCalls = False
        pythonCode = False
        finishReason = None
        for token in client.events():
            print(token.data)
            if stopGenerate:
                break
            if token.data != "[DONE]":
                choices = json.loads(token.data)['choices'][0]
                chunk = choices["text"]
                if choices["logprobs"] is not None:
                    finishReason = list(choices["logprobs"]['top_logprobs'][-1].keys())[-1]
                    print(finishReason)
                output += chunk
                if "<|p" in chunk:
                    toolCalls = True
                if "<|python_tag|>" in output and toolCalls:
                    try:
                        nextChar = output.split("<|python_tag|>")[-1][0]
                    except IndexError:
                        nextChar = '{'
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
        if stopGenerate:
            chat.append({'role': 'assistant', 'content': output+"<|eot_id|>"})
            userTurn = True
            break
        if output != "":
            userTurn = False if finishReason == '<|eom_id|>' else True
            suffix = "<|eom_id|>" if not userTurn else "<|eot_id|>"
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


@app.route('/sendtext', methods=['POST'])
def send_text():
    userText = request.get_json()['usertext']
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


@app.route('/stopgenerate')
def stop_generate():
    global stopGenerate
    stopGenerate = True
    return jsonify({'message': 'Generation stopped'}), 200


if __name__ == '__main__':
    print(f"Using model: {apimodel}")
    print(f"Parameter: {parameters}")
    # print(f"Access this: {public_url}")
    reset_conv()
    app.run(debug=False)
