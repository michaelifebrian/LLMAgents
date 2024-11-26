import json
import os
from flask import Flask, Response, stream_with_context, request, render_template, jsonify, send_from_directory
from tools import python_interpreter
from utils import query, generate_prompt, is_json
from prompt import system_prompt, toolsAlias, getFunction, parameters
# from pyngrok import ngrok
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
        print(prompt)
        try:
            parameters['stream'] = False
            data = query(URL, parameters, prompt).json()
        except Exception as e:
            yield f"error: {e}"
            break
        toolCalls = False
        finishReason = None
        choices = data['choices'][0]
        output = choices['text']
        if "<|python_tag|>" in output:
            toolCalls = True
            if output.split("<|python_tag|>")[-1][0] != "{":
                yield "**Running python code**\n\n"
                yield '```python\n'
                yield output.split("<|python_tag|>")[-1]
                yield '\n```\n'
        if not toolCalls:
            yield output
        if choices["logprobs"] is not None:
            finishReason = list(choices["logprobs"]['top_logprobs'][-1].keys())[-1]
            print(finishReason)
        yield '\n\n'
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
    # print(f"Access this: {public_url}")
    reset_conv()
    app.run(debug=False)
