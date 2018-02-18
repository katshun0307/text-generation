# -*- coding: utf-8 -*- #

from settings import *
import keras
import numpy as np
# import pickle as pkl


class DataGetter:

    def __init__(self):
        self.url = 'https://s3.amazonaws.com/text-datasets/nietzsche.txt'

    def setup(self):
        self.download()
        self.vectorize()

    def download(self):
        path = keras.utils.get_file('nietzsche.txt', origin=self.url)
        text = open(path).read().lower()
        print("Corpus length: %s" % len(text))
        self.text = text

    def vectorize(self):
        sentences = []
        next_chars = []

        for i in range(0, len(self.text) - maxlen, step):
            sentences.append(self.text[i: i + maxlen])
            next_chars.append(self.text[i + maxlen])
        print("number of sequences: %s" % len(sentences))

        chars = sorted(list(set(self.text)))  # SORT the LIST of unique char SETs in text
        print("Unique chars: %s" % len(chars))
        print(chars)
        char_indices = dict((char, chars.index(char)) for char in chars)

        # vectorization
        print("vectorization...")
        x = np.zeros((len(sentences), maxlen, len(chars)))
        y = np.zeros((len(sentences), len(chars)), dtype=np.bool)

        for i, sentence in enumerate(sentences):
            for t, char in enumerate(sentence):
                x[i, t, char_indices[char]] = 1
                y[i, char_indices[next_chars[i]]] = 1

        # # saving
        # print("saving...")
        # save_data = {'x': x, 'y': y}
        # with open('data.pkl', 'wb') as f:
        #     pkl.dump(save_data, f)
        # print("save success")
        print("data ready")

        self.x = x
        self.y = y
        self.sentences = sentences
        self.chars = chars
        self.char_indices = char_indices



if __name__ == '__main__':
    pass
