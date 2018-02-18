# -*- coding: utf-8 -*- #

from settings import *
from data.download import *
from keras.layers import *
import keras

class Network(keras.models.Sequential):
    def __init__(self, data: DataGetter):
        super().__init__()
        # model = keras.models.Sequential()

        self.add(LSTM(128, input_shape=(maxlen, len(data.chars))))
        self.add(Dense(len(data.chars), activation='softmax'))

        optimizer = keras.optimizers.RMSprop(lr=0.01)
        self.compile(loss='categorical_crossentropy', optimizer=optimizer)
        # self.model = model
        print("network compiled")

if __name__ == '__main__':
    data = DataGetter()
    data.setup()
    network = Network(data)
    print(type(network))
