#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
使用 LDA 算法重新实现一遍。
"""


import matplotlib.pyplot as plt
import numpy as np

import data_processing as dp
import model_selection as ms
from linear import LDA

if __name__ == '__main__':
    data = np.loadtxt('ex2data2.txt', delimiter=',')
    X = data[:, :2]
    y = data[:, 2:3]

    pos1 = (y == 0).ravel()
    pos2 = (y == 1).ravel()
    i1, = plt.plot(X[pos2, 0], X[pos2, 1], 'k+', linewidth=2, markersize=7)
    i2, = plt.plot(X[pos1, 0], X[pos1, 1], 'yo', markersize=7)
    plt.legend([i1, i2], ['y = 1', 'y = 0'], loc='upper right')
    plt.xlabel('Microchip Test 1')
    plt.ylabel('Microchip Test 2')
    plt.show()

    'Part 1: LDA classification'
    X = dp.map_features(X, degrees=6)
    print('映射后前5行：\n', X[:5, :])

    ld = LDA()
    ld.train(X, y)
    p = ld.predict(X)
    print('\nTrain Accuracy: %f' % (ms.accuracy(p, y) * 100))
