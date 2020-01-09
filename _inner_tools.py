#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
机器学习内部工具模块
"""

from typing import Union, Tuple

import numpy as np
from numpy import ndarray

from exception import DataNotMatchError


def r2m(row: ndarray) -> ndarray:
    """
    将行向量转成一行的矩阵。如果不是行向量就忽略。

    :param row: 行向量
    :return: 矩阵
    """

    if len(row.shape) == 1:
        row = row.reshape((1, row.shape[0]))

    return row


# def r2c(row: ndarray) -> ndarray:
#
#     if len(row.shape) == 1:
#         row = row.reshape((row.shape[0], 1))
#     elif len(row.shape) == 2 and row.shape[0] == 1:
#         row = row.reshape((row.shape[1], 1))
#
#     return row


def c2r(col: ndarray) -> ndarray:
    """
    将列向量转化为行向量。如果不是列向量就忽略。

    :param col: 列向量
    :return: 行向量
    """

    if len(col.shape) == 2 and col.shape[1] == 1:
        col = col.ravel()

    return col


def addones(x_mat: ndarray) -> ndarray:
    """
    检查 x_mat 是否有截距列，没有就加上。

    :param x_mat: 特征向量组，行数 m 表示样本数，列数 n 表示特征数
    :return: 处理完的 x_mat
    """

    x_mat = r2m(x_mat)
    if not (x_mat[:, 0] == 1).all():
        x_mat = np.hstack((np.ones((x_mat.shape[0], 1)), x_mat))

    return x_mat


def delones(x_mat: ndarray) -> Tuple[ndarray, bool]:
    """
    与 addones_x 作用相反，检查 x_mat 是否有截距列，有就去掉。

    :param x_mat: 特征向量组，行数 m 表示样本数，列数 n 表示特征数
    :return: 处理完的 x_mat；原 x_mat 中是否含有截距列
    """

    has_ones = False
    x_mat = r2m(x_mat)
    if (x_mat[:, 0] == 1).all():
        x_mat = x_mat[:, 1:]
        has_ones = True

    return x_mat, has_ones


def match_theta_x(theta: ndarray, x_mat: ndarray) -> Tuple[ndarray, ndarray]:
    """
    检查 x_mat 特征数是否和 theta_vec 一致，不一致给 x_mat 加上一列截距列。还是不一致抛出异常。
    注意此方法只能用在线性回归和逻辑回归中。

    :param theta: 参数向量，可以是列向量、行向量或者矩阵
    :param x_mat: 特征向量组，行数 m 表示样本数，列数 n 表示特征数
    :return: 检验完后的 x_mat；检验完后的 theta_vec
    """

    x_mat = r2m(x_mat)
    theta = c2r(theta)

    m = x_mat.shape[0]
    if x_mat.shape[1] + 1 == theta.shape[0]:
        x_mat = np.hstack((np.ones((m, 1)), x_mat))
    if x_mat.shape[1] != theta.shape[0]:
        raise DataNotMatchError('feature quantity mismatch')

    return theta, x_mat


def match_x_y(x_mat: ndarray, y_vec: ndarray, add_ones: bool = True) -> Tuple[ndarray, ndarray]:
    """
    检验 x_mat 和 y_vec 是否一致，不一致抛出异常。

    :param x_mat: 特征向量组，行数 m 表示样本数，列数 n 表示特征数
    :param y_vec: 输出向量，可以是列向量也可以是行向量，每一个值代表 x_mat 中对应行的输出
    :param add_ones: 是否给 x_mat 增加截距列
    :return: 检验完后的 x_mat；检验完后的 y_vec
    """

    if add_ones:
        x_mat = addones(x_mat)
    else:
        x_mat = r2m(x_mat)
    y_vec = c2r(y_vec)
    if len(y_vec.shape) == 0:
        if x_mat.shape[0] != 1:
            raise DataNotMatchError('number of samples does not match')
    elif x_mat.shape[0] != y_vec.shape[0]:
        raise DataNotMatchError('number of samples does not match')

    return x_mat, y_vec


def i2f_dtype(x: ndarray) -> ndarray:
    """
    将整数类型转换为对应的浮点数类型。

    :param x: 向量或矩阵
    :return: 类型转换后的向量或矩阵
    """

    # dtype 的不同会导致结果的不同。numpy 在计算 int 和 float 时，会把结果转成 int
    if x.dtype == np.int32:
        x = x.astype(np.float32)
    elif x.dtype == np.int_:
        x = x.astype(np.float_)
    elif x.dtype == np.int8 or x.dtype == np.int16:
        x = x.astype(np.float16)
    elif x.dtype == np.int64:
        x = x.astype(np.float64)

    return x


def u2i_dtype(x: ndarray) -> ndarray:
    """
    将无符号整数转换为有符号整数。

    :param x: 向量或矩阵
    :return: 类型转换后的向量或矩阵
    """

    if x.dtype == np.uint8:
        x = x.astype(dtype=np.int8)
    elif x.dtype == np.uint16:
        x = x.astype(dtype=np.int16)
    elif x.dtype == np.uint32:
        x = x.astype(dtype=np.int32)
    elif x.dtype == np.uint64:
        x = x.astype(dtype=np.int64)

    return x


def ret(r: ndarray) -> Union[int, float, ndarray]:
    """
    如果 r 中只有一个元素，返回那个元素，否则返回 r。

    :param r: ndarray 对象
    :return: 单个元素或 r
    """

    num = 1
    for size in r.shape:
        num *= size

    if num == 1:
        return r.ravel()[0]
    else:
        return r


def convert_y(labels: ndarray, y_row: ndarray, *, to: bool = True, to_labels: ndarray = np.array([1, 0])) -> ndarray:
    """
    to 是 True 的情况下，如果是二分类问题且 labels 不是 to_labels，就将它转换成 to_labels，否则无视。
    to 是 False 的情况下则相反，将 to_labels 转化为 labels。

    :param labels: 类别数组，包含所有类别的值，在二分类问题中，默认排在前面的类别是正类
    :param y_row: 输出行向量，取值是 labels 中的值
    :param to: 是否将 labels 转化为 to_labels
    :param to_labels: 要转化成的二分类标记
    :return: 转化后的 y_row
    """

    if len(labels) == 2 and not (labels == to_labels).all():
        if to:
            positive_idx = y_row == labels[0]
            negative_idx = y_row == labels[1]
            y_row = y_row.copy()
            y_row[positive_idx] = to_labels[0]
            y_row[negative_idx] = to_labels[1]
        else:
            positive_idx = y_row == to_labels[0]
            negative_idx = y_row == to_labels[1]
            y_row = y_row.copy()
            y_row[positive_idx] = labels[0]
            y_row[negative_idx] = labels[1]

    return y_row
