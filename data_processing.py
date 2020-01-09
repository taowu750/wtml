#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
这个模块是数据的处理模块。包括数据降维、特征规范化、特征映射等。
"""

from itertools import combinations
from itertools import repeat
from typing import Tuple

import numpy as np
from numpy import ndarray

import _inner_tools as __t
from exception import DataNotMatchError


def addones(x_mat: ndarray) -> ndarray:
    """
    检查 x_mat 是否有截距列，没有就加上。

    :param x_mat: 特征向量组，行数 m 表示样本数，列数 n 表示特征数
    :return: 处理完的 x_mat
    """

    if len(x_mat.shape) == 1:
        x_mat = x_mat.reshape((1, x_mat.shape[0]))
    if not (x_mat[:, 0] == 1).all():
        x_mat = np.hstack((np.ones((x_mat.shape[0], 1)), x_mat))

    return x_mat


def ndarray_size(*arrs: ndarray) -> int:
    """
    计算 ndarray 的元素数量。多个 ndarray 的话就返回全部元素数量的和。

    :param arrs: ndarray 数组
    :return: 元素数量
    """

    size = 0
    for a in arrs:
        num = 1
        for s in a.shape:
            num *= s
        size += num

    return size


def feature_normalize(x_mat: ndarray, mean_row: ndarray = None, std_row: ndarray = None) \
        -> Tuple[ndarray, ndarray, ndarray]:
    """
    将 x_mat 的特征规范化和归一化，也就是缩放以使得不同的参数之间差距不要太大。
    如果参数 mean_col 和 std_col 都提供了的话，就使用它们进行规范；否则就计算新的 mean_col 和 std_col。
    可以用在线性回归、SVM、PCA 中。
    截距列不会被规范化。

    :param x_mat: 特征向量组，行数 m 表示样本数，列数 n 表示特征数
    :param mean_row: 每列特征值的平均值行向量
    :param std_row: 每列特征值的标准行向量
    :return: 规范化后的特征向量组，原来的特征向量组不会被影响；每个特征的平均值向量；每个特征的标准差向量
    """

    x_mat, has_ones = __t.delones(x_mat)
    if mean_row is not None:
        mean_row = __t.c2r(mean_row)
    if std_row is not None:
        std_row = __t.c2r(std_row)
    n = x_mat.shape[1]
    if mean_row is not None and std_row is not None and \
            n != mean_row.shape[0] and mean_row.shape[0] != std_row.shape[0]:
        raise DataNotMatchError('x_mat\'s feature num does not match mean_row, std_row')

    # 转换数据类型防止计算精度丢失
    x_norm = __t.i2f_dtype(x_mat.copy())
    if mean_row is None and std_row is None:
        mean_row = np.mean(x_norm, axis=0)
        # ddof 设置为 1 表示计算样本方差，不设置表示计算总体方差
        std_row = np.std(x_norm, ddof=1, axis=0)
    x_norm = (x_norm - mean_row) / std_row
    if has_ones:
        x_norm = np.hstack((np.ones((x_norm.shape[0], 1)), x_norm))

    return x_norm, mean_row, std_row


def map_features(x_mat: ndarray, *, degrees: int = 3, variables: int = 2) -> ndarray:
    """
    将 x_mat 中的一次特征映射成多次特征。可以用在线性回归、逻辑回归中。
    注意不能包含截距列。

    :param x_mat: 特征向量组，行数 m 表示样本数，列数 n 表示特征数
    :param degrees: 最大次方数，也就是被选取特征次方之和最大值，取值需要大于等于 2
    :param variables: 最大乘项数，即每个乘项最多有多少个特征相乘。这个值不会大于 degree，也不会大于特征数。当它等于 1 时，就只取次方项
    :return: 映射完成的特征向量组
    """

    x_mat = __t.r2m(x_mat)
    m = x_mat.shape[0]
    n = x_mat.shape[1]
    if variables > degrees:
        variables = degrees
    if variables > n:
        variables = n

    out = np.ones((m, 1))
    if variables == 1:
        out = np.hstack((out, x_mat))
        for d in range(2, degrees + 1):
            for i in range(n):
                out = np.hstack((out, x_mat[:, i:i + 1] ** d))
    else:
        idx = list(range(n))
        degree_coms = []

        # 次方从 1 到 degree
        out = np.hstack((out, x_mat))
        for degree in range(2, degrees + 1):
            # 从 n 个特征中选出 x1,x2,...,x_variables
            for features_idx in list(combinations(idx, variables)):
                # 转置是因为在循环中，ndarray 是按行取值的。
                features = x_mat[:, features_idx].T
                # 获取次方数的组合
                degree_com = list(repeat(0, variables))
                __degree_combinations(degree, variables, degree_com, degree_coms)
                for degree_c in degree_coms:
                    val = 1
                    for f, dc in zip(features, degree_c):
                        val *= 1 if dc == 0 else f ** dc
                    out = np.hstack((out, val.reshape((val.shape[0], 1))))
                degree_coms.clear()

    return out


def __degree_combinations(degree: int, num: int, com: list, coms: list):
    """
    列举所有在最大次方为 degree，有 num 个特征条件下，所有次方的组合。

    :param degree: 最大次方和
    :param num: 特征数
    :param com: 一次次方组合的列表
    :param coms: 存储所有次方组合的列表
    :return: 次方组合列表
    """

    if num == 1:
        com[-1] = degree
        coms.append(tuple(com))
    else:
        for d in reversed(range(degree + 1)):
            com[-num] = d
            __degree_combinations(degree - d, num - 1, com, coms)
