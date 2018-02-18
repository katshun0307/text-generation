# -*- coding: utf-8 -*- #

from network import Network
from data.download import *
from settings import *
from predict import sample
import random
import sys
import numpy as np
import time

class Trainer:

    def __init__(self, network: Network, data: DataGetter):
        self.network = network
        self.data = data

    def train(self):
        network = self.network
        data = self.data
        for epoch in range(1, 60):
            print("epoch: %s" % epoch)
            network.fit(data.x, data.y, batch_size=128, epochs=1)
            timestr = time.strftime("%Y%m%d")
            network.save('./models/' + timestr + '_' + str(epoch) + '.h5')
            network.save_weights('./models/weights_' + timestr + '_' + str(epoch) + '.h5')

            # test generate
            start_index = random.randint(0, len(data.text) - maxlen - 1)
            generated_text = data.text[start_index: start_index + maxlen]
            print('---generating with seed: "' + generated_text + '"')

            for temp in [0.2, 0.5, 1.0, 1.2]:
                print("\n-----temperature %s-------\n" % temp)
                sys.stdout.write(generated_text)

                for i in range(400):
                    sampled = np.zeros((1, maxlen, len(data.chars)))
                    for t, char in enumerate(generated_text):
                        sampled[0, t, data.char_indices[char]] = 1

                    preds = network.predict(sampled, verbose=0)[0]
                    next_index = sample(preds, temp)
                    next_char = data.chars[next_index]

                    generated_text += next_char
                    generated_text = generated_text[1:]

                    sys.stdout.write(next_char)


if __name__ == '__main__':
    data = DataGetter()
    data.setup()
    model = Network(data)
    trainer = Trainer(model, data)
    trainer.train()
    # x, y, sentences, chars, char_indices = vectorize(text)
    # train(model, x, y, text, char_indices, chars)
