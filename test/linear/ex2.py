#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
在这部分练习中，您将建立一个logistic回归模型来预测学生是否被大学录取。
假设你是一所大学的系主任，你想根据每个申请者在两次考试中的成绩来决定他们的入学机会。
您有以前申请者的历史数据，可以用作逻辑回归的训练集。对于每个培训示例，您都有申请者在两次
考试中的分数和入学决定。
你的任务是建立一个分类模型，根据这两次考试的分数来估计申请者的录取概率。这个大纲和ex2.py
中的框架代码将指导您完成这个练习。
"""

import matplotlib.pyplot as plt
import numpy as np

import model_selection as ms
from linear import LogisticReg

if __name__ == '__main__':
    data = np.loadtxt('ex2data1.txt', delimiter=',')
    # 必须要 :2 才能切除列向量
    X = data[:, :2]
    y = data[:, 2]

    'Part 1: Plotting'
    print('Plotting data with + indicating (y = 1) examples and o indicating (y = 0) examples.')
    i1, = plt.plot(X[y == 1, 0], X[y == 1, 1], 'k+', linewidth=2, markersize=7)
    i2, = plt.plot(X[y == 0, 0], X[y == 0, 1], 'ko', markerfacecolor='y', markersize=7)
    plt.legend([i1, i2], ['Admitted', 'Not admitted'], loc='upper right')
    plt.xlabel('Exam 1 score')
    plt.ylabel('Exam 2 score')
    plt.show()

    'Part 3: Optimizing using fminunc'
    # theta = gr.train(X, y)
    lo = LogisticReg()
    X = X.astype(dtype=float)
    theta = lo.train(X, y)

    # TODO: 绘制决策边界

    'Part 4: Predict and accuracies'
    # prob = mf.sigmoid(np.array([1, 45, 85]) @ theta)
    prob = lo.probability(np.array([1, 45, 85]))
    print('\nFor a student with scores 45 and 85, we predict an admission probability of %f' % prob)
    print('Expected value: 0.775 +/- 0.002\n')

    # p = gr.predict(theta, X)
    p = lo.predict(X)
    print('Train Accuracy: %f' % (ms.accuracy(p, y) * 100))
    print('Expected accuracy (approx): 89.0')

    'Part5: 查准率和召回率'

    precisions, recalls, f1, *_ = ms.precision_and_recall(p, y)
    print('\n查准率是 %.2f，召回率是 %.2f，f1 值是 %.2f' % (precisions[1], recalls[1], f1[1]))

    'Part6: 绘制 P-R 曲线'

    precisions, recalls = ms.pr(lo.probability(X), y)
    ms.plot_pr(precisions, recalls)

    'Part7: 绘制 ROC 曲线'

    tprs, fprs, auc = ms.roc(lo.probability(X), y)
    ms.plot_roc(tprs, fprs)
    print('auc = %.3f' % auc)

    'Part8: 计算代价敏感错误率'

    # 我们假定让原本能通过的人不过的代价要更大
    cost_mat = {0: {0: 0, 1: 1},
                1: {0: 5, 1: 0}}
    cost_sensitive = ms.cost_sensitive(p, y, cost_mat)
    print('\n代价敏感错误率：', cost_sensitive)
