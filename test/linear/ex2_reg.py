#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
在练习的这一部分中，您将实现正则化逻辑回归，以预测来自制造工厂的微芯片是否通过质量保证（QA）。
在质量保证期间，每个微芯片都要经过各种测试，以确保其正常工作。假设你是工厂的产品经理，
你有两个不同测试的一些微芯片的测试结果。从这两个测试中，您想确定是否应该接受或拒绝微芯片。
为了帮助你做出决定，你有一个过去微芯片测试结果的数据集，从中你可以建立一个逻辑回归模型。

一种更好地拟合数据的方法是从每个数据点创建更多的特征。在提供的函数mapfeature()中，
我们将把这些特征映射到x1和x2的所有多项式项，直到第六次方。
作为这个映射的结果，我们的两个特征向量（两个qa测试的分数）被转换成一个28维向量。
在这个高维特征向量上训练的logistic回归分类器将具有更复杂的决策边界，并且在我们的二维图
中绘制时将呈现非线性。
"""

import matplotlib.pyplot as plt
import numpy as np

import data_processing as dp
import model_selection as ms
from linear import LogisticReg


def map_feature(x1, x2):
    degree = 6
    # 在这里添加了偏置列
    out = np.ones((x1.shape[0], 1))
    # 从 1 次方到 degree 次方
    for i in range(1, degree + 1):
        # x1 和 x2 次方之和不超过 i
        for j in range(0, i + 1):
            out = np.hstack((out, (x1 ** (i - j)) * (x2 ** j)))

    return out


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

    'Part 1: Regularized Logistic Regression'
    X = dp.map_features(X, degrees=6)
    X = dp.addones(X)
    print('映射后前5行：\n', X[:10, :])
    initial_lambda = 1
    initial_theta = np.zeros(X.shape[1])
    lo = LogisticReg(lamb=initial_lambda)
    lo.train(X, y)
    p = lo.predict(X)
    print('\nTrain Accuracy: %f' % (ms.accuracy(p, y) * 100))
    print('Expected accuracy (with lambda = 1): 83.1 (approx)')

    # TODO: 绘制决策边界
