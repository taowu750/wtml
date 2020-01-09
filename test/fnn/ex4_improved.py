#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
使用改进的前馈神经网络训练手写数字的例子。
"""

import numpy as np
import scipy.io as scio

import model_selection as ms
from neural_network import FNN
from .display_data import display_data

if __name__ == '__main__':
    'Part1: Loading and Visualizing data'

    # data 是一个 dict，里面存有 .mat 里面的矩阵数据
    data = scio.loadmat('ex4data1.mat')
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

    print('\nTraining Feedforward Neutral Network...')
    split = 4000
    # 随机打乱数据非常重要
    X = X[rand_indices, :]
    y = y[rand_indices]
    Xtrain = X[:split, :]
    ytrain = y[:split]
    Xcv = X[split:, :]
    ycv = y[split:]

    fnn = FNN(labels=list(range(1, num_labels + 1)), layer_nodes=[400, 25, 10], lamb=1, max_iter=50)
    fnn.train(Xtrain, ytrain)
    p = fnn.predict(Xcv)
    print('Training set accuracy: %f\n' % (np.mean(p == ycv) * 100))

    'Part4: 查准率和召回率'

    precisions, recalls, f1, *_ = ms.precision_and_recall(p, ycv, labels=list(range(1, num_labels + 1)))
    for label in precisions:
        print('标签 %d: 查准率是 %.2f，召回率是 %.2f，f1 值是 %.2f' % (label, precisions[label], recalls[label], f1[label]))

    'Part5: P-R 图'

    # 查看数字 3 的 P-R 图
    precisions, recalls, *_ = ms.pr(fnn.probability(Xcv)[:, 2], ycv, positive_label=3)
    ms.plot_pr(precisions, recalls)

    'Part6: ROC 图'

    tprs, fprs, auc = ms.roc(fnn.probability(Xcv)[:, 2], ycv, positive_label=3)
    ms.plot_roc(tprs, fprs, title='number 3')
    print('\nAUC =', auc, '\n')

    # 'Part6: 学习曲线'
    #
    # cost_train, cost_cv = ms.cost_curve(Xtrain, ytrain, Xcv, ycv,
    #                                     lambda: FNN(labels=list(range(1, num_labels + 1)),
    #                                                 layer_nodes=[400, 25, 10], lamb=1, max_iter=50),
    #                                     add_ones=False, ran=list(range(250, 5001, 250)))
    # ms.plot_cost_curve(cost_train, cost_cv, title='Learning Curve', ran=list(range(250, 5001, 250)))
