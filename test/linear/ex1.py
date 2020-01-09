#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
在整个练习中，您将使用脚本ex1.py和ex1 multi.py。
这些脚本为问题设置数据集并调用要编写的函数。你不需要修改它们中的任何一个。您只需要按照此分配中
的说明修改其他文件中的函数。对于这个编程练习，您只需要完成练习的第一部分，就可以用一个变量
实现线性回归。练习的第二部分是可选的，包括多变量线性回归。

假设你是一家连锁餐厅的首席执行官，正在考虑在不同的城市开设一家新的分店。这家连锁店在各个城市
都有卡车，你有数据显示城市的利润和人口。您希望使用此数据帮助您选择下一个要展开的城市。
文件ex1data1.txt包含线性回归问题的数据集。第一列是一个城市的人口，第二列是该城市食品卡车
的利润。负值表示亏损。
"""

import matplotlib.pyplot as plt
import numpy as np

from linear import LinearReg

if __name__ == '__main__':
    'Part2: Plotting'

    print('Plotting Data ...')
    data = np.loadtxt('ex1data1.txt', delimiter=',')

    X = data[:, :1]
    y = data[:, 1:2]
    m = data.shape[0]

    plt.xlabel('Population of City in 10,000s')
    plt.ylabel('Profit in $10,000s')
    plt.plot(X, y, 'rx', markersize=6)
    plt.show()

    print('\nRunning Gradient Descent ...')
    max_iter = 1500
    li = LinearReg(max_iter=max_iter)
    theta = li.train(X, y)
    print('theta found by gradient descent:')
    print(theta)
    print('Expected theta values (approx):')
    print(' -3.6303\n  1.1664')

    # Plot the linear fit
    i1, = plt.plot(X[:, 0], y, 'rx', markersize=6)
    i2, = plt.plot(X[:, 0], li.predict(X), 'b-')
    plt.legend([i1, i2], ['Training data', 'Linear regression'], loc='upper right')
    plt.show()

    predict1 = li.predict(np.array([1, 3.5]))
    print('For population = 35,000, we predict a profit of %f' % (predict1 * 10000))
    predict2 = li.predict(np.array([1, 7]))
    print('For population = 70,000, we predict a profit of %f' % (predict2 * 10000))

    'Part 4: Visualizing J(theta_0, theta_1)'
    # TODO: 绘制直观的 3d 图像
