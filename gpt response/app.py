from flask import Flask, render_template, request, jsonify
import torch
import random
import numpy as np
import json
from model import NeuralNet, bag_of_words, stem, tokenize

app = Flask(__name__)

with open('Hindi101.json', 'r', encoding="utf8") as f:
    intents = json.load(f)

FILE = "TrainData.pth"
data = torch.load(FILE)

input_size = data['input_size']
hidden_size = data['hidden_size']
output_size = data['output_size']
all_words = data['all_words']
tags = data['tags']
model_state = data['model_state']

model = NeuralNet(input_size, hidden_size, output_size)
model.load_state_dict(model_state)
model.eval()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get')
def get_bot_response():
    userText = request.args.get('msg')
    if userText == "":
        return ""
    sentence = tokenize(userText)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(dtype=torch.float32)

    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    for intent in intents['intents']:
        if intent['tag'] == tag:
            response = random.choice(intent['responses'])

    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)
