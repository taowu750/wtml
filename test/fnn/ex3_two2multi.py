#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
测试 Two2MultiClassLearner 在手写数字数据集上的表现。
"""

import numpy as np
import scipy.io as scio

import model_selection as ms
from linear import LogisticReg

if __name__ == '__main__':
    'Part1: Loading and Visualizing data'

    # data 是一个 dict，里面存有 .mat 里面的矩阵数据
    data = scio.loadmat('ex3data1.mat')
    X = data['X']
    y = data['y'].ravel()
    num_labels = 10
    m = X.shape[0]

    rand_indices = np.arange(0, 5000)
    np.random.shuffle(rand_indices)

    print('\nTraining One-vs-All Logistic Regression...')
    lamb = 0.1
    split = 4000
    # 随机打乱数据非常重要
    X = X[rand_indices, :]
    y = y[rand_indices]
    Xtrain = X[:split, :]
    ytrain = y[:split]
    Xcv = X[split:, :]
    ycv = y[split:]

    tm = ms.Two2MultiClassLearner(
        learner=lambda positive_label, negative_label: LogisticReg(lamb=lamb, labels=[positive_label, negative_label]),
        labels=list(range(1, num_labels + 1)), strategy='ovr')
    # tm = ms.Two2MultiClassLearner(
    #     learner=lambda positive_label, negative_label: LDA(labels=[positive_label, negative_label]),
    #     labels=list(range(1, num_labels + 1)), strategy='ovr')
    tm.train(Xtrain, ytrain)
    p = tm.predict(Xcv)
    print('Training set accuracy: %f' % (np.mean(p == ycv) * 100))

    'Part4: 查准率和召回率'

    precisions, recalls, f1, *_ = ms.precision_and_recall(p, ycv, labels=list(range(1, num_labels + 1)))
    print()
    for label in precisions:
        print('标签 %d: 查准率是 %.2f，召回率是 %.2f，f1 值是 %.2f' % (label, precisions[label], recalls[label], f1[label]))

    'Part5: P-R 图'

    # 查看数字 3 的 P-R 图
    precisions, recalls, *_ = ms.pr(tm.probability(Xcv)[:, 2], ycv, positive_label=3)
    ms.plot_pr(precisions, recalls)

    'Part6: ROC 图'

    tprs, fprs, auc = ms.roc(tm.probability(Xcv)[:, 2], ycv, positive_label=3)
    ms.plot_roc(tprs, fprs, title='number 3')
    print('\nAUC =', auc)

    # 'Part7: 学习曲线'
    #
    # cost_train, cost_cv = ms.cost_curve(Xtrain, ytrain, Xcv, ycv,
    #                                     lambda: ms.Two2MultiClassLearner(
    #                                         learner=lambda positive_label, negative_label: LogisticReg(lamb=lamb),
    #                                         labels=list(range(1, num_labels + 1)), positive_label=1,
    #                                         negative_label=0, strategy='ovr'),
    #                                     add_ones=False, ran=list(range(250, 5001, 250)))
    # ms.plot_cost_curve(cost_train, cost_cv, title='Learning Curve', ran=list(range(250, 5001, 250)))
