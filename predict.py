# -*- coding: utf-8 -*- #

import numpy as np


def sample(preds, temp=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temp
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)

    return np.argmax(probas)


if __name__ == '__main__':
    pass
