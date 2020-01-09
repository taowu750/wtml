#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
包含了一些简单的线性算法的模块，包括线性回归、逻辑回归和线性判别分析算法
"""

from collections import OrderedDict
from typing import Union, Tuple, List

import numpy as np
import scipy.optimize as opt
from numpy import ndarray

import _inner_tools as _t
import data_processing as _dp
import mlfunc as _mf
from base import ISuperviseLearner, IConfidenceLearner, IProbabilityLearner
from exception import StateError, DataNotMatchError


class LinearReg(ISuperviseLearner):
    """
    线性回归学习器，用于回归任务。
    """

    def __init__(self, *, lamb: Union[int, float] = 0, max_iter: int = 100, method: str = 'gradient',
                 mean_row: Union[ndarray, None] = None, std_row: Union[ndarray, None] = None):
        """
        初始化线性回归。

        :param lamb: 正则化参数，默认为 0
        :param max_iter: 训练的最大迭代次数，默认为 100
        :param method: 训练使用的方法，有 gradient（梯度下降优化法）和 normal（正规方程）
        :param mean_row: 每列特征值的平均值行向量
        :param std_row: 每列特征值的标准差行向量
        """

        self.lamb = lamb
        self.max_iter = max_iter
        self.method = method
        self.mean_row = mean_row
        self.std_row = std_row

        self.x_mat = None
        self.y_row = None
        self._theta_row = None

    @property
    def method(self) -> str:
        return self._method

    @method.setter
    def method(self, value: str):
        if value != 'gradient' and value != 'normal':
            raise ValueError('method must be "gradient" or "normal')

        self._method = value

    @property
    def lamb(self) -> Union[int, float]:
        return self._lamb

    @lamb.setter
    def lamb(self, value: Union[int, float]):
        if value < 0:
            raise ValueError('lamb must be greater than or equal to 0')
        self._lamb = value

    @property
    def max_iter(self) -> int:
        return self._max_iter

    @max_iter.setter
    def max_iter(self, value: int):
        if value < 1:
            raise ValueError('max_iter must be greater than or equal to 1')
        self._max_iter = value

    @property
    def theta_row(self) -> Union[ndarray, None]:
        return self._theta_row

    def train(self, x_mat: ndarray, y_row: ndarray) -> ndarray:
        """
        进行训练。可以使用梯度下降或正规方程算法进行训练，其中正规方程法对于特征不是很多的情况下，
        比如 n <= 10000，会取得很好的效果。

        :param x_mat: 特征向量组，行数 m 表示样本数，列数 n 表示特征数
        :param y_row: 输出行向量，每一个值代表 x_mat 中对应行的输出
        :return 训练的来的参数
        """

        x_mat, y_row = _t.match_x_y(x_mat, y_row)
        n = x_mat.shape[1]

        if self.method == 'gradient':
            self._theta_row = opt.fmin_cg(f=lambda t, x, y: self.__cost(t, x, y),
                                          x0=np.zeros((n,)), args=(x_mat, y_row), maxiter=self.max_iter,
                                          fprime=lambda t, x, y: self.__gradient(t, x, y))
        elif self.method == 'normal':
            self._theta_row = self.__normal_eqn(x_mat, y_row)
        else:
            raise ValueError('parameter method must be "gradient" or "normal')

        self.x_mat = x_mat
        self.y_row = y_row

        return self._theta_row

    def predict(self, x_mat: ndarray):
        """
        使用训练好的参数进行预测。如果提供了特征缩放时的平均值、标准差向量，那么会先将特征值规范化。

        :param x_mat: 特征向量组，行数 m 表示样本数，列数 n 表示特征数
        :return: 预测结果
        """

        if self._theta_row is None:
            raise StateError('not trained yet')

        self._theta_row, x_mat = _t.match_theta_x(self._theta_row, x_mat)
        # 正规方程法不需要规范化数据
        if self.method == 'gradient' and self.mean_row is not None and self.std_row is not None:
            x_mat = _dp.feature_normalize(x_mat, mean_row=self.mean_row, std_row=self.std_row)[0]

        return _t.ret(x_mat @ self._theta_row)

    def cost(self, x_mat: ndarray, y_row: ndarray):
        """
        计算在 x_mat 和 y_row 上的代价。

        :param x_mat: 特征向量组，行数 m 表示样本数，列数 n 表示特征数
        :param y_row: 输出行向量，每一个值代表 x_mat 中对应行的输出
        :return: 代价值
        """

        if self._theta_row is None:
            raise StateError('not trained yet')

        self._theta_row, x_mat = _t.match_theta_x(self._theta_row, x_mat)
        x_mat, y_row = _t.match_x_y(x_mat, y_row)

        return self.__cost(self._theta_row, x_mat, y_row)

    def __cost(self, theta_row: ndarray, x_mat: ndarray, y_row: ndarray) -> float:
        """
        计算线性代价函数值 J。

        :param theta_row: 参数行向量，每一行/列都是 x_mat 中对应特征的参数
        :param x_mat: 特征向量组，行数 m 表示样本数，列数 n 表示特征数
        :param y_row: 输出行向量，每一个值代表 x_mat 中对应行的输出
        :return: 代价值
        """

        m = x_mat.shape[0]
        thetan = theta_row[1:]

        return sum((x_mat @ theta_row - y_row) ** 2) / (2 * m) + self.lamb * sum(thetan ** 2) / (2 * m)

    def __gradient(self, theta_row: ndarray, x_mat: ndarray, y_vec: ndarray) -> ndarray:
        """
        计算梯度。

        :param theta_row: 参数行向量，每一行/列都是 x_mat 中对应特征的参数
        :param x_mat: 特征向量组，行数 m 表示样本数，列数 n 表示特征数
        :param y_vec: 输出向量，可以是列向量也可以是行向量，每一个值代表 x_mat 中对应行的输出
        :return: 梯度，是一个行向量。
        """

        m = x_mat.shape[0]
        thetan = theta_row[1:]
        xn = x_mat[:, 1:]
        hx = x_mat @ theta_row

        grad0 = ((x_mat[:, :1].T @ (hx - y_vec)) / m).ravel()
        grad1 = ((xn.T @ (hx - y_vec)) / m + self.lamb * thetan / m).ravel()

        return np.hstack((grad0, grad1))

    def __normal_eqn(self, x_mat: ndarray, y_vec: ndarray) -> ndarray:
        """
        使用正规方程计算参数向量。对于特征不是很多的情况下，比如 n <= 10000，此时它会取得很好的效果。
        注意 x_mat 不要有偏置列。

        :param x_mat: 特征向量组，行数 m 表示样本数，列数 n 表示特征数
        :param y_vec: 输出向量，可以是列向量也可以是行向量，每一个值代表 x_mat 中对应行的输出
        :param lamb: 正则化参数，默认为 0
        :return: 参数向量
        """

        regularization = 0
        if self.lamb != 0:
            regularization = np.eye(x_mat.shape[1])
            regularization[0, 0] = 0

        return np.linalg.pinv(x_mat.T @ x_mat + self.lamb * regularization) @ x_mat.T @ y_vec


class LogisticReg(IProbabilityLearner):
    """
    逻辑回归学习器，用于分类问题。
    """

    def __init__(self, *, lamb: Union[int, float] = 0, max_iter: int = 100,
                 labels: Union[None, Tuple[int], List[int], ndarray] = None, threshold: float = 0.5,
                 mean_row: Union[ndarray, None] = None, std_row: Union[ndarray, None] = None):
        """
        初始化线性回归。

        :param lamb: 正则化参数，默认为 0
        :param max_iter: 训练的最大迭代次数，默认为 100
        :param labels: 类别数组，包含所有类别的值，为 None 表示训练的是二分类问题，且类别标记为 1, 0。
                       在二分类问题中，默认排在前面的类别是正类
        :param threshold: 阈值，默认为 0.5，用在二分类问题中
        :param mean_row: 每列特征值的平均值行向量
        :param std_row: 每列特征值的标准差行向量
        """

        self.lamb = lamb
        self.max_iter = max_iter
        self.labels = labels
        self.threshold = threshold
        self.mean_row = mean_row
        self.std_row = std_row

        self.x_mat = None
        self.y_vec = None
        self._theta = None

    @property
    def lamb(self) -> Union[int, float]:
        return self._lamb

    @lamb.setter
    def lamb(self, value: Union[int, float]):
        if value < 0:
            raise ValueError('lamb must be greater than or equal to 0')
        self._lamb = value

    @property
    def max_iter(self) -> int:
        return self._max_iter

    @max_iter.setter
    def max_iter(self, value: int):
        if value < 1:
            raise ValueError('max_iter must be greater than or equal to 1')
        self._max_iter = value

    @property
    def threshold(self) -> float:
        return self._threshold

    @threshold.setter
    def threshold(self, value: float):
        if value < 0 or value > 1:
            raise ValueError('threshold must be between 0 and 1')
        self._threshold = value

    @property
    def labels(self) -> ndarray:
        return self._labels

    @labels.setter
    def labels(self, value: Union[ndarray, Tuple[int], List[int], None]):
        if value is None:
            value = np.array([1, 0])
        if len(value) < 2:
            raise ValueError('labels contains at least two classes')

        if not isinstance(value, ndarray):
            self._labels = np.asarray(value)
        else:
            self._labels = value

    @property
    def theta(self) -> Union[ndarray, None]:
        return self._theta

    # TODO: 将面向列的数据转化成面向行的数据
    def train(self, x_mat: ndarray, y_row: ndarray) -> ndarray:
        """
        训练逻辑回归，既可以训练二分类问题，也可以训练多类分类问题。

        :param x_mat: 特征向量组，行数 m 表示样本数，列数 n 表示特征数
        :param y_row: 输出行向量，每一个值代表 x_mat 中对应行的输出
        :return: 参数。二分类问题返回行向量。多类分类问题返回 n*num_labels 的矩阵，每一列表示对应类别的参数。
        """

        x_mat, y_row = _t.match_x_y(x_mat, y_row)
        y_row = _t.convert_y(self.labels, y_row)
        n = x_mat.shape[1]

        if len(self.labels) == 2:
            self._theta = opt.fmin_cg(f=lambda t, x, y: self.__cost(t, x, y),
                                      x0=np.zeros((n,)), args=(x_mat, y_row), maxiter=self.max_iter,
                                      fprime=lambda t, x, y: self.__gradient(t, x, y))
        else:
            self._theta = np.zeros((n, len(self.labels)))

            for i, label in enumerate(self.labels):
                self._theta[:, i] = opt.fmin_cg(f=lambda t, x, y: self.__cost(t, x, y == label),
                                                x0=np.zeros((n,)), args=(x_mat, y_row), maxiter=self.max_iter,
                                                fprime=lambda t, x, y: self.__gradient(t, x, y == label))

        return self._theta

    def predict(self, x_mat: ndarray) -> Union[ndarray, int]:
        """
        返回预测值，是对应于 x_mat 的标记。

        :param x_mat: 特征向量组，行数 m 表示样本数，列数 n 表示特征数
        :return: 预测标记
        """

        if self._theta is None:
            raise StateError('not trained yet')

        self._theta, x_mat = _t.match_theta_x(self._theta, x_mat)
        prob = x_mat @ self._theta

        if len(self.labels) == 2:
            return _t.ret(_t.convert_y(self.labels, _mf.sigmoid(prob) >= self.threshold, to=False))
        else:
            return _t.ret(self.labels[np.argmax(prob, axis=1)])

    def cost(self, x_mat: ndarray, y_row: ndarray) -> float:
        """
        计算在 x_mat 和 y_row 上的代价。

        :param x_mat: 特征向量组，行数 m 表示样本数，列数 n 表示特征数
        :param y_row: 输出行向量，每一个值代表 x_mat 中对应行的输出
        :return: 代价值。
        """

        if self._theta is None:
            raise StateError('not trained yet')

        self._theta, x_mat = _t.match_theta_x(self._theta, x_mat)
        x_mat, y_row = _t.match_x_y(x_mat, y_row)

        if len(self.labels) == 2:
            return self.__cost(self._theta, x_mat, y_row)
        else:
            m = x_mat.shape[0]
            cost_sum = 0
            for i, label in enumerate(self.labels):
                y = y_row == label
                cost_sum = cost_sum + np.sum(y) * self.__cost(self._theta[:, i], x_mat, y) / m

            return cost_sum

    def probability(self, x_mat: ndarray) -> Union[ndarray, float]:
        """
        返回对应于 x_mat 的预测概率。如果是二分类问题，那么返回一个行向量；如果是多分类问题，返回一个
        m*num_labels 的矩阵，其中每一行表示样本在每个类上的概率。

        :param x_mat: 特征向量组，行数 m 表示样本数，列数 n 表示特征数
        :return: 预测概率。
        """

        if self._theta is None:
            raise StateError('not trained yet')

        self._theta, x_mat = _t.match_theta_x(self._theta, x_mat)

        return _mf.sigmoid(x_mat @ self._theta)

    def __cost(self, theta_row: ndarray, x_mat: ndarray, y_row: ndarray) -> float:
        """
        计算代价值。

        :param theta_row: 参数
        :param x_mat: 特征向量组，行数 m 表示样本数，列数 n 表示特征数
        :param y_row: 输出行向量，每一个值代表 x_mat 中对应行的输出
        :return: 代价值
        """

        m = x_mat.shape[0]
        hx = _mf.sigmoid(x_mat @ theta_row)
        thetan = theta_row[1:]
        # FIXME: 当 hx 中包含 0 或 1 时，会导致 log 出现警告并终止迭代
        cost = -(y_row @ np.log(hx) + (1 - y_row) @ np.log(1 - hx)) / m + self.lamb * sum(thetan ** 2) / (2 * m)

        return cost

    def __gradient(self, theta_row: ndarray, x_mat: ndarray, y_row: ndarray) -> ndarray:
        """
        计算梯度。

        :param theta_row: 参数
        :param x_mat: 特征向量组，行数 m 表示样本数，列数 n 表示特征数
        :param y_row: 输出行向量，每一个值代表 x_mat 中对应行的输出
        :return: 梯度，是一个行向量
        """

        m = x_mat.shape[0]
        thetan = theta_row[1:]
        xn = x_mat[:, 1:]
        hx = _mf.sigmoid(x_mat @ theta_row)

        # 如果不展开的话，就是个只有一个元素的二维数组
        grad0 = ((x_mat[:, :1].T @ (hx - y_row)) / m).ravel()
        gradn = ((xn.T @ (hx - y_row)) / m + self.lamb * thetan / m).ravel()

        return np.hstack((grad0, gradn))


class LDA(IConfidenceLearner):
    """
    线性判别分析（Linear Discriminant Analysis, LDA）学习器，可以进行分类，还可以对数据进行降维。
    """

    def __init__(self, *, labels: Union[None, Tuple[int], List[int], ndarray] = None):
        """
        初始化 LDA。

        :param labels: 类别数组，包含所有类别的值，为 None 表示训练的是二分类问题，且类别标记为 1, 0。
                       在二分类问题中，默认排在前面的类别是正类
        """

        self.labels = labels

        self._theta = None
        self._center_projections = {}

    @property
    def labels(self) -> ndarray:
        return self._labels

    @labels.setter
    def labels(self, value: Union[ndarray, Tuple[int], List[int], None]):
        if value is None:
            value = np.array([1, 0])
        if len(value) < 2:
            raise ValueError('labels contains at least two classes')

        if not isinstance(value, ndarray):
            self._labels = np.asarray(value)
        else:
            self._labels = value

    @property
    def theta(self) -> Union[ndarray, None]:
        return self._theta

    def train(self, x_mat: ndarray, y_row: ndarray):
        """
        使用给定的数据对 LDA 学习算法进行训练。

        :param x_mat: 特征向量组，行数 m 表示样本数，列数 n 表示特征数
        :param y_row: 输出行向量，每一个值代表 x_mat 中对应行的输出
        :return: 学习得来的参数
        """

        x_mat, y_row = _t.match_x_y(x_mat, y_row, add_ones=False)
        self._center_projections.clear()

        if len(self.labels) == 2:
            # 两种类别的样本
            x0 = x_mat[y_row == self.labels[0], :]
            x1 = x_mat[y_row == self.labels[1], :]

            # 样本中心点
            u0 = np.mean(x0, axis=0)
            u1 = np.mean(x1, axis=0)

            sw = (x0 - u0).T @ (x0 - u0) + (x1 - u1).T @ (x1 - u1)  # 类内散度矩阵

            self._theta = np.linalg.pinv(sw) @ (u0 - u1)
            self._center_projections[self.labels[0]] = self._theta @ u0
            self._center_projections[self.labels[1]] = self._theta @ u1
        else:
            xn = []
            un = OrderedDict()
            u_mean = 0
            for label in self.labels:
                xn.append(x_mat[y_row == label, :])
                un[label] = np.mean(xn[-1], axis=0)
                u_mean = u_mean + un[label]
            u_mean = u_mean / x_mat.shape[0]

            sw = 0
            sb = 0
            n = u_mean.shape[0]
            for x, u in zip(xn, un.values()):
                sw = sw + (x - u).T @ (x - u)
                sb = sb + x.shape[0] * (u - u_mean).reshape((n, 1)) @ (u - u_mean).reshape((1, n))

            ev, fv = np.linalg.eig(np.linalg.pinv(sw) @ sb)
            # 去掉 0 特征
            fv = fv[:, np.isclose(ev, 0) == False]
            ev = ev[np.isclose(ev, 0) == False]
            # 从大到小排序
            # TODO: 考虑是否只在压缩的时候进行排序
            indices = np.argsort(ev)[::-1]
            fv = fv[:, indices]

            self._theta = np.real(fv)
            for label in self.labels:
                self._center_projections[label] = self._theta.T @ un[label]

        return self._theta

    def predict(self, x_mat: ndarray) -> Union[int, ndarray]:
        """
        返回预测值，是对应于 x_mat 的标记。

        :param x_mat: 特征向量组，行数 m 表示样本数，列数 n 表示特征数
        :return: 预测标记
        """

        x_mat = self.__match_theta_x(x_mat)
        result_labels = np.empty((x_mat.shape[0],))

        x_mat = x_mat @ self._theta
        for i, x in enumerate(x_mat):
            r = None
            min_dist = None
            for label, center in self._center_projections.items():
                dist = np.sum((x - center) ** 2)
                if min_dist is None or min_dist > dist:
                    min_dist = dist
                    r = label
            result_labels[i] = r

        return _t.ret(result_labels)

    def cost(self, x_mat: ndarray, y_row: ndarray) -> float:
        """
        计算在 x_mat 和 y_row 上的代价。

        :param x_mat: 特征向量组，行数 m 表示样本数，列数 n 表示特征数
        :param y_row: 输出行向量，每一个值代表 x_mat 中对应行的输出
        :return: 代价值。
        """

        x_mat = self.__match_theta_x(x_mat)
        x_mat, y_row = _t.match_x_y(x_mat, y_row, add_ones=False)

        if len(self.labels) == 2:
            # 两种类别的样本
            x0 = x_mat[y_row == self.labels[0], :]
            x1 = x_mat[y_row == self.labels[1], :]

            # 样本中心点
            u0 = np.mean(x0, axis=0)
            u1 = np.mean(x1, axis=0)

            n = u0.shape[0]
            sw = (x0 - u0).T @ (x0 - u0) + (x1 - u1).T @ (x1 - u1)  # 类内散度矩阵
            sb = (u0 - u1).reshape((n, 1)) @ (u0 - u1).reshape((1, n))  # 类间散度矩阵

            # 和书上分子分母相反，只是因为约定代价值越小越好，而原来的公式是越大越好，故取反
            return (self._theta @ sw @ self._theta) / (self._theta @ sb @ self._theta)
        else:
            xn = []
            un = OrderedDict()
            u_mean = 0
            for label in self.labels:
                xn.append(x_mat[y_row == label, :])
                un[label] = np.mean(xn[-1], axis=0)
                u_mean = u_mean + un[label]
            u_mean = u_mean / x_mat.shape[0]

            sw = 0
            sb = 0
            n = u_mean.shape[0]
            for x, u in zip(xn, un.values()):
                sw = sw + (x - u).T @ (x - u)
                sb = sb + x.shape[0] * (u - u_mean).reshape((n, 1)) @ (u - u_mean).reshape((1, n))

            return np.trace(self._theta.T @ sw @ self._theta) / np.trace(self._theta.T @ sb @ self._theta)

    def confidence(self, x_mat: ndarray):
        """
        返回对 x_mat 的预测置信度。如果是二分类问题，那么返回一个行向量；如果是多分类问题，
        返回一个 m*num_labels 的矩阵，其中每一行表示样本在每个类上的置信度。

        :param x_mat: 特征向量组，行数 m 表示样本数，列数 n 表示特征数
        :return: 置信度
        """

        x_mat = self.__match_theta_x(x_mat)
        x_mat = x_mat @ self._theta

        if len(self._labels) == 2:
            confidences = np.empty((x_mat.shape[0],))
            center = tuple(self._center_projections.values())[0]
            for i, x in enumerate(x_mat):
                confidences[i] = np.exp(-np.sum((x - center) ** 2))

            return _t.ret(confidences)
        else:
            confidences = np.empty((x_mat.shape[0], len(self._labels)))
            for i, x in enumerate(x_mat):
                for j, center in enumerate(self._center_projections.values()):
                    confidences[i][j] = np.sum((x - center) ** 2)

            return confidences

    def reduce(self, x_mat: ndarray, dimension: int) -> ndarray:
        """
        对 x_mat 进行降维，返回降维后的数据。需要注意的是，所能降维的范围为 [1, 类别数-1]。

        :param x_mat: 特征向量组，行数 m 表示样本数，列数 n 表示特征数
        :param dimension: 新的维度，这个维度必须小于等于类别数。
        :return: 降维后的矩阵
        """

        x_mat = self.__match_theta_x(x_mat)
        if dimension < 1 or dimension > len(self.labels) - 1:
            raise ValueError('dimension range is [1, class number - 1]')

        if len(self.labels) == 2:
            return (x_mat @ self._theta).reshape((x_mat.shape[0], 1))
        else:
            return x_mat @ self._theta[:, :dimension]

    def restore(self, reduced_mat: ndarray) -> ndarray:
        """
        将数据从降维中还原。

        :param reduced_mat: 降维后的矩阵
        :return: 还原的矩阵。
        """

        if self._theta is None:
            raise StateError('not trained yet')

        if len(reduced_mat.shape) != 2:
            raise ValueError('reduced_mat must be a matrix')

        dimension = reduced_mat.shape[1]
        if dimension > len(self.labels) - 1:
            raise ValueError('reduced_mat is not a compressed matrix')

        if len(self.labels) == 2:
            return reduced_mat @ np.linalg.pinv(self._theta.reshape((self._theta.shape[0], 1)))
        else:
            return reduced_mat @ np.linalg.pinv(self._theta[:, :dimension])

    def __match_theta_x(self, x_mat: ndarray) -> ndarray:
        """
        检查输入是否和参数匹配。

        :param x_mat: 特征向量组，行数 m 表示样本数，列数 n 表示特征数
        :return:
        """

        if self._theta is None:
            raise StateError('not trained yet')

        x_mat = _t.r2m(x_mat)
        if x_mat.shape[1] != self._theta.shape[0]:
            raise DataNotMatchError('feature quantity mismatch')

        return x_mat
