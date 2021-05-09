import tensorflow as tf
import argparse
from flask import Flask, render_template, request
import sys
from chatting import chatbot_tf1 as chbot

sys.path.append("./models/")
app = Flask(__name__)
app.static_folder = 'static'

# set the hyper-paramters
parser = argparse.ArgumentParser(description='GPT-2 chatbot')
parser.add_argument('--nsamples', type=int, default=1)

parser.add_argument('--top_k', type=int, default=40)

parser.add_argument('--top_p', type=int, default=0.7)

parser.add_argument('--temperature', type=float, default=1.0)

parser.add_argument('--batch_size', type=int, default=1)

parser.add_argument('--length', type=int, default=400)

args = parser.parse_args()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    return chbot.interact_model(
        text=userText,
        nsamples=args.nsamples,
        top_k=args.top_k,
        top_p=args.top_p,
        temperature=args.temperature,
        batch_size=args.batch_size,
        length=args.length)

if __name__ == "__main__":
    app.run()
