# -*- coding: utf-8 -*- #

import numpy as np


def reweight_distribution(original_distribution, temp=0.5):
    """
    higher temps result in unexpected results
    :param original_distribution: softmax distribution
    :param temp: temperature
    :return:  new distribution
    """
    distribution = np.log(original_distribution) / temp
    distribution = np.exp(distribution)
    return distribution / np.sum(distribution)

if __name__ == '__main__':
    pass
