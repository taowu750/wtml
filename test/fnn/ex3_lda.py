#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
使用 LDA 算法重写识别手写数字的例子。
"""


import numpy as np
import scipy.io as scio

import model_selection as ms
from linear import LDA
from .display_data import display_data

if __name__ == '__main__':
    'Part1: Loading and Visualizing data'

    # data 是一个 dict，里面存有 .mat 里面的矩阵数据
    data = scio.loadmat('ex3data1.mat')
    X = data['X']
    y = data['y'].ravel()
    num_labels = 10
    m = X.shape[0]

    # 随机选取 100 个图片进行显示
    rand_indices = np.arange(0, 5000)
    np.random.shuffle(rand_indices)
    sel = X[rand_indices[:100], :]
    display_data(sel)

    'Part3: One-vs-all training'

    print('\nTraining One-vs-All Logistic Regression...')
    split = 4000
    # 随机打乱数据非常重要
    X = X[rand_indices, :]
    y = y[rand_indices]
    Xtrain = X[:split, :]
    ytrain = y[:split]
    Xcv = X[split:, :]
    ycv = y[split:]

    ld = LDA(labels=list(range(1, num_labels + 1)))
    ld.train(Xtrain, ytrain)
    p = ld.predict(Xcv)
    print('Training set accuracy: %f' % (np.mean(p == ycv) * 100))

    'Part4: 查准率和召回率'

    precisions, recalls, f1, *_ = ms.precision_and_recall(p, ycv, labels=list(range(1, num_labels + 1)))
    for label in precisions:
        print('标签 %d: 查准率是 %.2f，召回率是 %.2f，f1 值是 %.2f' % (label, precisions[label], recalls[label], f1[label]))

    'Part5: 压缩还原数据'

    print('\n压缩前前5行数据')
    print(Xcv[:5, :])

    reduced = ld.reduce(Xcv, 3)
    print('\n压缩后的 Shape 和前 5 行数据')
    print(reduced.shape)
    print(reduced[:5, :])

    print('\n还原后的 Shape 和前 5 行数据')
    restored = ld.restore(reduced)
    print(restored.shape)
    print(restored[:5, :])

    'Part6: 学习曲线'

    cost_train, cost_cv = ms.cost_curve(Xtrain, ytrain, Xcv, ycv, lambda: LDA(labels=np.arange(1, num_labels + 1)),
                                        add_ones=False, ran=list(range(200, 5001, 200)))
    ms.plot_cost_curve(cost_train, cost_cv, title='Learning Curve', ran=list(range(200, 5001, 200)))
