#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
在这一部分中，您将实现多变量线性回归来预测房价。假设你在卖房子，你想知道一个好的市场价格是多少。
一种方法是首先收集最近售出的房子的信息，并制作一个房价模型。
文件ex1data2.txt包含俄勒冈州波特兰市房价的培训集。第一栏是房子的大小（平方英尺），第二栏是
卧室的数量，第三栏是房子的价格。ex1_multi.py脚本已设置为帮助您逐步完成此练习。
"""

import numpy as np

from data_processing import feature_normalize
from linear import LinearReg
from model_selection import linear_error

if __name__ == '__main__':
    'Part1: Feature Normalization'

    data = np.loadtxt('ex1data2.txt', delimiter=',')
    X = data[..., :2]
    y = data[..., 2:]
    m = data.shape[0]

    print('First 10 examples from the dataset: ')
    for i in range(10):
        print(' x = [%.0f %.0f], y = %.0f' % (X[i, 0], X[i, 1], y[i]))

    print('\nNormalizing Features ...')
    X, mean, std = feature_normalize(X)
    print('10 examples from featureNormalize(X):')
    for i in range(10):
        print('x = [%.0f %.0f]' % (X[i, 0], X[i, 1]))

    # Add intercept term to X
    X = np.hstack((np.ones((m, 1)), X))

    'Part2: Gradient Descent'

    print('\nRunning gradient descent ...\n')
    alpha = 0.1
    num_iter = 300

    li = LinearReg(max_iter=num_iter)
    theta = li.train(X, y)
    print('Theta computed from gradient descent:')
    print(theta)
    print('The mean squared error from gradient descent:')
    li.mean_row = mean
    li.std_row = std
    tmp = li.predict(X)
    for i in range(10): print(tmp[i], y[i])
    print(linear_error(tmp, y), '\n')

    # plt.plot([i for i in range(1, J_history.shape[0] + 1)], J_history, '-b', linewidth=2)
    # plt.xlabel('Number of iterations')
    # plt.ylabel('Cost J')
    # plt.show()

    price = li.predict(np.array([1650, 3]))
    print('Predicted price of a 1650 sq-ft, 3 br house (using gradient descent):\n $%f' % price)

    'Part3: Normal Equations'

    print('\nSolving with normal equations...')
    data = np.loadtxt('ex1data2.txt', delimiter=',')
    X = data[..., :2]
    y = data[..., 2:]

    # X = np.hstack((np.ones((m, 1)), X))
    # theta = lr.normal_eqn(X, y)
    li.method = 'normal'
    theta = li.train(X, y)
    print('Theta computed from the normal equations: ')
    print(theta)

    price = li.predict(np.array([1650, 3]))
    print('Predicted price of a 1650 sq-ft, 3 br house (using normal equation):\n $%f' % price)
