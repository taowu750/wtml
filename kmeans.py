#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import Iterable

import numpy as np
from numpy import ndarray

import _inner_tools as __t


def find_closest(x_mat: ndarray, centroids_mat: ndarray):
    """
    k均值算法中，对每个样本找到其对应最接近的聚类中心的下标。

    :param x_mat: 特征向量组，行数 m 表示样本数，列数 n 表示特征数
    :param centroids_mat: 聚类中心
    :return: 样本对应聚类中心下标行向量
    """

    x_mat = __t.r2m(x_mat)
    centroids_mat = __t.r2m(centroids_mat)

    k = centroids_mat.shape[0]
    m = x_mat.shape[0]
    idx = np.empty((m,), dtype=int)

    for i in range(m):
        min_j = 0
        min_dist = np.sum((x_mat[i, :] - centroids_mat[0, :]) ** 2)
        for j in range(1, k):
            dist = np.sum((x_mat[i, :] - centroids_mat[j, :]) ** 2)
            if dist < min_dist:
                min_dist = dist
                min_j = j
        idx[i] = min_j

    return idx


def compute_centroids(x_mat: ndarray, idx_vec: ndarray):
    """
    根据之前的聚类中心下标和样本，计算移动后的平均聚类中心值。

    :param x_mat: 特征向量组，行数 m 表示样本数，列数 n 表示特征数
    :param idx_vec: 聚类中心下标
    :return: 新的聚类中心
    """

    x_mat = __t.r2m(x_mat)
    idx_vec = __t.c2r(idx_vec)

    m, n = x_mat.shape
    k = np.max(idx_vec) + 1
    centroids = np.zeros((k, n))
    cen_num = np.zeros((k,))

    for i in range(m):
        centroids[idx_vec[i], :] = centroids[idx_vec[i], :] + x_mat[i, :]
        cen_num[idx_vec[i]] = cen_num[idx_vec[i]] + 1
    for i in range(k):
        if cen_num[i] > 0:
            centroids[i, :] = centroids[i, :] / cen_num[i]

    return centroids


def train(x_mat: ndarray, k: int, *, max_iters: int = 10, initial_centroids: Iterable = None, history: bool = False):
    """
    进行k均值训练

    :param x_mat: 特征向量组，行数 m 表示样本数，列数 n 表示特征数
    :param k: 聚类数目
    :param max_iters: 最大迭代次数
    :param initial_centroids: 初始聚类中心，不提供别的话将随机挑选聚类中心
    :param history: 是否返回历史信息
    :return: 计算好的聚类中心；包含每个样本所属聚类中心下标的行向量；包含每一次迭代计算的聚类中心列表（history为True的话）
    """

    x_mat = __t.r2m(x_mat)

    m, n = x_mat.shape
    if initial_centroids is None:
        rand_indices = np.arange(0, m)
        np.random.shuffle(rand_indices)
        initial_centroids = x_mat[rand_indices[:k], :]
    if not isinstance(initial_centroids, ndarray):
        initial_centroids = np.asarray(initial_centroids)

    idx = None
    centroids_history = None
    if history:
        centroids_history = [initial_centroids]
    for i in range(max_iters):
        idx = find_closest(x_mat, initial_centroids)
        initial_centroids = compute_centroids(x_mat, idx)
        if history:
            centroids_history.append(initial_centroids)

    if history:
        return initial_centroids, idx, centroids_history
    else:
        return initial_centroids, idx
