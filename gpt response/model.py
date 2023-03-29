import torch
import nltk
import numpy as np

from nltk.stem import SnowballStemmer

stemmer = SnowballStemmer('english')

def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, all_words):
    tokenized_sentence = [stem(w) for w in tokenized_sentence]
    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1.0
    return bag

class NeuralNet(torch.nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNet, self).__init__()
        self.hidden = torch.nn.Linear(input_size, hidden_size)
        self.hidden2 = torch.nn.Linear(hidden_size, hidden_size)
        self.output = torch.nn.Linear(hidden_size, output_size)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        out = self.hidden(x)
        out = self.relu(out)
        out = self.hidden2(out)
        out = self.relu(out)
        out = self.output(out)
        return out

    def load_state_dict(self, state_dict):
        # Rename the keys of the state dictionary to match the model's key names
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('l1'):
                new_key = 'hidden' + key[2:]
                new_state_dict[new_key] = value
            elif key.startswith('l2'):
                new_key = 'hidden2' + key[2:]
                new_state_dict[new_key] = value
            elif key.startswith('l3'):
                new_key = 'output' + key[2:]
                new_state_dict[new_key] = value
            else:
                raise KeyError('Unexpected key found in state_dict: {}'.format(key))

        # Call the original load_state_dict function with the renamed keys
        super(NeuralNet, self).load_state_dict(new_state_dict)

