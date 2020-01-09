#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
这个模块包含了一些常用的函数，例如激活函数、计算信息增益等。
"""


import math
from typing import Union

import numpy as np
from numpy import ndarray


def sigmoid(x: Union[ndarray, int, float]) -> Union[float, ndarray]:
    """
    sigmoid 激励函数。

    :param x: 激励函数变量，可以是 ndarray 或数字类型
    :return: 激励函数值，类型与输入变量相同
    """

    return 1 / (1 + np.exp(-x))


def sigmoid_gradient(x: Union[ndarray, int, float]) -> Union[float, ndarray]:
    """
    计算在 x 值处的 sigmoid 函数的导数

    :param x: 输入值
    :return: sigmoid 梯度
    """

    hx = sigmoid(x)
    return hx * (1 - hx)


def ent(y_row: ndarray) -> float:
    """
    计算信息熵，它的值越小，表示 y_row 的纯度越高。

    :param y_row: 输出向量。是一个只有一个维度的行向量
    :return: 信息熵
    """

    m = y_row.shape[0]
    pk = {}
    for y in y_row:
        if y in pk:
            pk[y] = pk[y] + 1
        else:
            pk[y] = 1
    for y, count in pk.items():
        pk[y] = count / m

    inf = 0
    for p in pk.values():
        inf = inf - p * (0 if math.isclose(p, 0) else math.log2(p))

    return inf


def gini(y_row: ndarray) -> float:
    """
    计算基尼值，它的值越小，表示 y_row 的纯度越高。

    :param y_row: 输出向量。是一个只有一个维度的行向量
    :return: 基尼值
    """

    m = y_row.shape[0]
    pk = {}
    for y in y_row:
        if y in pk:
            pk[y] = pk[y] + 1
        else:
            pk[y] = 1
    for y, count in pk.items():
        pk[y] = count / m

    inf = 1
    for p in pk.values():
        inf = inf - p ** 2

    return inf


def gaussian_kernel(x1: Union[ndarray, int, float], x2: Union[ndarray, int, float], gamma: float) -> float:
    """
    高斯核函数。

    :param x1: 一个特征向量
    :param x2: 另一个特征向量
    :param gamma: 核函数的参数，即核函数的带宽，超圆的半径。
    :return: 取值在 [0, 1] 区间上
    """

    return np.exp(-np.sum((x1 - x2) ** 2) / (2 * gamma ** 2))
