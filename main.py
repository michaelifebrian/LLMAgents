import json
import os
import transformers
from flask import Flask, Response, stream_with_context, request, render_template, jsonify, send_from_directory
from tools import searchengine, scrapeweb, stable_diffusion_generate_image, pythoninterpreter, searchengineduckduckgo
from utils import parseoutput, query
import markdown

getFunction = {
    "searchengine": searchengine,
    "scrapeweb": scrapeweb,
    "stable_diffusion_generate_image": stable_diffusion_generate_image,
    "pythoninterpreter": pythoninterpreter,
    # "RAGPDF": RAGPDF
}
TOOLS = [stable_diffusion_generate_image,
         scrapeweb,
         searchengine,
         pythoninterpreter,
         # RAGPDF
         ]
toolsAlias = {
    "stable_diffusion_generate_image": "Stable Diffusion Image Generator",
    "scrapeweb": "Web Scrapper",
    "searchengine": "Web Search",
    "pythoninterpreter": "Python Interpreter"
}
stream = True
streammaxtoken = 150
model = "Qwen/Qwen2.5-72B-Instruct"
URL = "https://api-inference.huggingface.co/models/Qwen/Qwen2.5-72B-Instruct"
seed = int.from_bytes(os.urandom(8), 'big')
max_tokens = 8000
system_prompt = "You are qwen Chatbot. You has been placed in this agent with all" \
                "features by Michael. Always introduce yourself first. Michael using you for his portfolio. You are his" \
                "projects. People may ask about Michael's including his other projects, his interest, current academics," \
                " and other. Michael will give his identity and his information below. Also, you are helpful assistant, " \
                "that can help other people task beside you giving information about Michael."
chat = []
userturn = None
tokenizer = transformers.AutoTokenizer.from_pretrained(model)


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
        prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True, tools=TOOLS)
        fulloutput = ""
        data = ' '
        dataprint = ''
        istoolcall = False
        while data != '':
            try:
                data = query(URL,
                             {"model": model,
                              "prompt": prompt,
                              "max_tokens": streammaxtoken,
                              'seed': seed,
                              })
                try:
                    data = data['choices'][0]["text"]
                    prompt += data
                    fulloutput += data
                except:
                    print(data)
                    break
                if "<tool_call>" in data and "</tool_call>" not in data:
                    dataprint = data.split("<tool_call>")[0].rstrip()
                    istoolcall = True
                if "<tool_call>" not in data and "</tool_call>" in data:
                    dataprint = data.split("</tool_call>")[1].lstrip()
                    istoolcall = False
                if "<tool_call>" in data and "</tool_call>" in data:
                    dataprint = data.split("<tool_call>")[0].rstrip() + "\n\n" + data.split("</tool_call>")[1].lstrip()
                    istoolcall = False
                if "<tool_call>" not in data and "</tool_call>" not in data and not istoolcall:
                    dataprint = data
                    istoolcall = False
                if "<tool_call>" not in data and "</tool_call>" not in data and istoolcall:
                    dataprint = ''
                    istoolcall = True
                print(dataprint, end="")
                yield dataprint
            except:
                break
        output = fulloutput
        print("\n", end="")
        yield "\n"
        parsed = parseoutput(output)
        if parsed[0] != []:
            userturn = False
            tool_calls = []
            for i in range(len(parsed[0])):
                tool_calls.append({
                    'type': 'function',
                    'function': {
                        'name': parsed[0][i],
                        'arguments': json.dumps(parsed[1][i])
                    }
                })
            chat.append({'role': 'assistant', 'content': parsed[2], 'tool_calls': tool_calls})
            yield "\n"
            for i in range(len(parsed[0])):
                yield f"**{toolsAlias[parsed[0][i]]} called.**"
                yield "\n\n"
                chat.append({
                    'role': 'tool',
                    'name': parsed[0][i],
                    'content': getFunction[parsed[0][i]](**parsed[1][i])
                })
        elif parsed[0] == [] and parsed[2] != "ERROR":
            userturn = True
            chat.append({'role': 'assistant', 'content': parsed[2]})
        elif parsed[0] == [] and parsed[2] == "ERROR":
            userturn = False
            chat.append({'role': 'assistant', 'content': 'ERROR PARSING JSON', 'tool_calls': 'ERROR PARSING JSON'})
        if '<|STOP|>' in output:
            return


app = Flask(__name__)


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


@app.route('/chat_tool_history')
def tools_history():
    return markdown.markdown(tokenizer.apply_chat_template(chat, tokenize=False, tools=TOOLS).replace("\n", "\n\n"))


@app.route('/chatdict')
def chat_history():
    return jsonify(chat)


if __name__ == '__main__':
    # resetconv()
    # app.run(host='0.0.0.0', port=5000, )
    print(searchengineduckduckgo("michael"))
