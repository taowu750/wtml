#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from typing import Union, List, Tuple, Callable

import numpy as np
from numpy import ndarray

import _inner_tools as _t
from base import IConfidenceLearner
from exception import DataNotMatchError, StateError
from mlfunc import gaussian_kernel


# TODO: 把双下划线减去一个
# TODO: 改变 None 值作为默认值
# TODO: 学习器必须经过严密的参数检查和状态检查
# TODO: 为学习器添加接收权值的功能，使得它们能够从文件中获取权值
class SVC(IConfidenceLearner):
    __KERNEL = {'gauss', 'linear'}

    """
    支持向量机分类器，用于二分类问题，默认使用 1 作为正例，-1 作为负例。
    如果想要扩展到多分类器，请使用 Two2MultiClassLearner。
    """

    def __init__(self, *, labels: Union[ndarray, List[int], Tuple[int]] = np.array([1, -1]), c: float = 1.0,
                 kernel: Union[str, Callable[[Union[ndarray, int, float],
                                              Union[ndarray, int, float]], float]] = 'gauss',
                 gamma: float = 0, tol: float = 1e-3, max_iter: int = 40):
        """
        初始化支持向量机。

        :param labels: 类别数组，包含所有类别的值，默认为 1, -1。排在前面的类别是正类
        :param c: 正则化系数。C越大，对误分类的惩罚增大，这样会出现训练集测试时准确率很高，但容易导致过拟合。 C值小，
                  对误分类的惩罚减小，容错能力增强，泛化能力较强，但也可能欠拟合。默认为 1
        :param kernel: 核函数类型，默认有 gauss 和 linear 两种。gauss 核函数适用于样本比较复杂的情况，linear 核函数适用于
                       样本线性可分的情况。默认为 gauss
        :param gamma: 核函数系数，用于 gauss 核函数，表示它的带宽。gamma 越小则越有可能过拟合，反之则会欠拟合。默认为 0 表示
                      使用特征数的倒数作为 gamma 值
        :param tol: 残差收敛条件，默认是0.001，这用来判断训练参数是否收敛
        :param max_iter: 最大迭代次数，默认是 40。
        """

        self.labels = labels
        self.c = c
        self.kernel = kernel
        self.gamma = gamma
        self.tol = tol
        self.max_iter = max_iter

        self._alphas = None
        self._b = None
        self._errors = None
        self._x_mat = None
        self._y_row = None
        self._theta = None

    @property
    def labels(self) -> ndarray:
        return self._labels

    @labels.setter
    def labels(self, value: Union[ndarray, Tuple[int], List[int]]):
        if len(value) != 2:
            raise ValueError('labels only contains two classes')

        if not isinstance(value, ndarray):
            self._labels = np.asarray(value)
        else:
            self._labels = value

    @property
    def c(self) -> float:
        return self._c

    @c.setter
    def c(self, value: float):
        if value < 0:
            raise ValueError('C must be greater than 0')

        self._c = value

    @property
    def kernel(self) -> Union[str, Callable[[Union[ndarray, int, float], Union[ndarray, int, float]], float]]:
        return self._kernel

    @kernel.setter
    def kernel(self, value: Union[str, Callable[[Union[ndarray, int, float], Union[ndarray, int, float]], float]]):
        if isinstance(value, str) and value not in self.__KERNEL:
            raise ValueError('kernel must be a function or take the value in', self.__KERNEL)

        self._kernel = value

    @property
    def gamma(self) -> float:
        return self._gamma

    @gamma.setter
    def gamma(self, value: float):
        if value < 0:
            raise ValueError('gamma must be greater than 0')

        self._gamma = value

    @property
    def tol(self) -> float:
        return self._tol

    @tol.setter
    def tol(self, value: float):
        if value < 0 or value > 1:
            raise ValueError('tol must be in [0, 1]')

        self._tol = value

    @property
    def max_iter(self) -> int:
        return self._max_iter

    @max_iter.setter
    def max_iter(self, value: int):
        if value < 1:
            raise ValueError('max_iter must be greater than 0')
        self._max_iter = value

    @property
    def theta(self):
        return self._theta

    def train(self, x_mat: ndarray, y_row: ndarray):
        """


        :param x_mat:
        :param y_row:
        :return:
        """

        x_mat, y_row = _t.match_x_y(x_mat, y_row, add_ones=False)
        y_row = _t.u2i_dtype(y_row)
        y_row = _t.convert_y(self.labels, y_row, to_labels=np.array([1, -1]))

        m, n = x_mat.shape
        if self.kernel == 'linear':
            k_mat = x_mat @ x_mat.T
        elif self.kernel == 'gauss':
            if not self.gamma:
                self.gamma = 1 / n
            k_mat = np.empty((m, m), dtype=np.float)
            for i, xi in enumerate(x_mat):
                for j in range(i, m):
                    k_mat[i, j] = gaussian_kernel(xi, x_mat[j], self.gamma)
                    k_mat[j, i] = k_mat[i, j]
        else:
            k_mat = np.empty((m, m), dtype=np.float)
            for i, xi in enumerate(x_mat):
                for j in range(i, m):
                    k_mat[i, j] = self.kernel(xi, x_mat[j])
                    k_mat[j, i] = k_mat[i, j]
        self._alphas = np.zeros((m,), dtype=np.float)
        self._b = 0
        self._errors = self._calc_errors(k_mat, y_row)
        print('params initialized')

        iter_num = 0
        iter_all = True
        changed_num = 0
        while (iter_num < self.max_iter) and (changed_num > 0 or iter_all):
            changed_num = 0
            # 在整个样本集和非边界样本集之间切换
            if iter_all:
                for i1 in range(m):
                    # 运行 smo 过程
                    changed_num = changed_num + self._smo(k_mat, y_row, i1)
            else:
                invalid_list = np.argwhere((self._alphas <= 0) | (self._alphas >= self.c)).ravel() if self._c else \
                    np.argwhere(self._alphas <= 0).ravel()
                for i1 in invalid_list:
                    # 运行 smo 过程
                    changed_num = changed_num + self._smo(k_mat, y_row, i1)
            if iter_all:
                iter_all = False
            elif changed_num == 0:
                iter_all = True
            iter_num = iter_num + 1
            print(iter_num, ',', changed_num, ':', self._alphas[self._alphas > 0])
        # 只使用支持向量构造划分超平面法向量和其他参数
        idx = self._alphas > 0
        self._x_mat = x_mat[idx, :]
        self._y_row = y_row[idx]
        self._alphas = self._alphas[idx]
        self._theta = self._alphas * self._y_row @ self._x_mat

    def predict(self, x_mat: ndarray):
        """
        返回预测值，是对应于 x_mat 的标记。

        :param x_mat: 特征向量组，行数 m 表示样本数，列数 n 表示特征数
        :return: 预测标记
        """

        if self._theta is None:
            raise StateError('not trained yet')
        x_mat = _t.r2m(x_mat)
        if x_mat.shape[1] != self._theta.shape[0]:
            raise DataNotMatchError('feature quantity mismatch')

        if self.kernel == 'linear':
            pred = x_mat @ self._theta + self._b
        elif self.kernel == 'gauss':
            m = x_mat.shape[0]
            pred = np.empty((m,))
            for i in range(m):
                for j in range(self._x_mat.shape[0]):
                    pred[i] = pred[i] + self._alphas[j] * self._y_row[j] * gaussian_kernel(x_mat[i], self._x_mat[j],
                                                                                           self.gamma)
                pred[i] = pred[i] + self._b
        else:
            m = x_mat.shape[0]
            pred = np.empty((m,))
            for i in range(m):
                for j in range(self._x_mat.shape[0]):
                    pred[i] = pred[i] + self._alphas[j] * self._y_row[j] * self.kernel(x_mat[i], self._x_mat[j])
                pred[i] = pred[i] + self._b

        return _t.ret(_t.convert_y(self.labels, (pred >= 0).astype(dtype=np.int16), to=False))

    def cost(self, x_mat: ndarray, y_row: ndarray) -> float:
        pass

    def confidence(self, x_mat: ndarray):
        pass

    def _smo(self, k_mat: ndarray, y_row: ndarray, i1: int) -> bool:
        """
        SMO（Sequential Minimal Optimization）算法，用来高效地优化支持向量机。

        :param k_mat: 核函数生成矩阵
        :param y_row: 输出集
        :param i1: 第一个变量下标
        :return: 此变量是否被改变
        """

        y1 = y_row[i1]
        e1 = self._errors[i1]
        alpha1old = self._alphas[i1]
        # 选择违反 KKT 条件的 alphai 作为第一个变量
        condition = (self.c and ((y1 * e1 < -self.tol and alpha1old < self.c) or
                                 (y1 * e1 > self.tol and alpha1old > 0))) or \
                    (not self.c and (y1 * e1 > self.tol and alpha1old > 0))
        if condition:
            # 选择预测误差相差最大的 alphai 作为第二个变量
            i2 = self._e2idx(i1)
            y2 = y_row[i2]
            e2 = self._errors[i2]
            alpha2old = self._alphas[i2]
            iota = k_mat[i1, i1] + k_mat[i2, i2] - 2 * k_mat[i1, i2]
            if iota > 0:
                alpha2new = alpha2old + y2 * (e1 - e2) / iota
                # 根据 alpha2new 的取值范围选取正确的 alpha2new
                low, high = self._low_high(y1, y2, alpha1old, alpha2old)
                # 如果上下限相同，则重新选择第一个变量
                if self.c and np.isclose(low, high):
                    return False
                if self.c and alpha2new > high:
                    alpha2new = high
                elif alpha2new < low:
                    alpha2new = low
            else:
                # 此时是临界情况，需要在边界处找极值点
                low, high = self._low_high(y1, y2, alpha1old, alpha2old)
                # 如果上下限相同，则重新选择第一个变量
                if self.c and np.isclose(low, high):
                    return False
                if self.c:
                    s = y1 * y2
                    low1 = alpha1old + s * (alpha2old - low)
                    high1 = alpha1old + s * (alpha2old - high)

                    f1 = y1 * (e1 - self._b) - alpha1old * k_mat[i1, i1] - s * alpha2old * k_mat[
                        i1, i2]
                    f2 = y2 * (e2 - self._b) - s * alpha1old * k_mat[i1, i2] - alpha2old * k_mat[
                        i2, i2]
                    psi_l = low1 * f1 + low * f2 + low1 ** 2 * k_mat[i1, i1] / 2 + low ** 2 * k_mat[
                        i2, i2] / 2 + s * low * low1 * k_mat[i1, i2]
                    psi_h = high1 * f1 + high * f2 + high1 ** 2 * k_mat[i1, i1] / 2 + high ** 2 * k_mat[
                        i2, i2] / 2 + s * high * high1 * k_mat[i1, i2]

                    # 选取边界处较小的值
                    alpha2new = low if psi_l < psi_h else high
                else:
                    alpha2new = low
            # 如果选取的变量不能带来明显的改变，则重新选取第一个变量
            if abs(alpha2new - alpha2old) < 0.00001:
                return False
            # 由 alpha2new 得出 alpha1new
            alpha1new = alpha1old + y1 * y2 * (alpha2old - alpha2new)
            # 求 b
            b_old = self._b
            b1new = -e1 - y1 * k_mat[i1, i1] * (alpha1new - alpha1old) - y2 * k_mat[i2, i1] * (
                    alpha2new - alpha2old) + b_old
            b2new = -e2 - y1 * k_mat[i1, i2] * (alpha1new - alpha1old) - y2 * k_mat[i2, i2] * (
                    alpha2new - alpha2old) + b_old
            # 更新模型参数
            if (self.c and 0 < alpha1new < self.c) or (not self.c and alpha1new > 0):
                self._b = b1new
            elif (self.c and 0 < alpha2new < self.c) or (not self.c and alpha2new > 0):
                self._b = b2new
            else:
                self._b = (b1new + b2new) / 2
            self._alphas[i1] = alpha1new
            self._alphas[i2] = alpha2new

            # 更新预测误差
            self._errors = self._calc_errors(k_mat, y_row)

            return True

        return False

    def _calc_errors(self, k_mat: ndarray, y_row: ndarray) -> ndarray:
        return self._b + self._alphas * y_row @ k_mat.T - y_row

    def _e2idx(self, i: int) -> int:
        """
        选取和 i 处预测误差差别最大的误差的下标

        :param i: 误差下标
        :return: 差别最大的误差的下标
        """

        e1 = self._errors[i]
        idx = 0 if i != 0 else 1
        e2 = self._errors[idx]

        if e1 > 0:
            for j, e in enumerate(self._errors):
                if j == i:
                    continue
                if e < e2:
                    e2 = e
                    idx = j
        else:
            for j, e in enumerate(self._errors):
                if j == i:
                    continue
                if e > e2:
                    e2 = e
                    idx = j

        return idx

    def _low_high(self, y1: float, y2: float, alpha1old: float, alpha2old: float) -> Tuple[float, float]:
        """
        返回在约束条件下 alpha2new 的最小和最大取值。

        :param y1:
        :param y2:
        :param alpha1old:
        :param alpha2old:
        :return:
        """

        if y1 != y2:
            low = max([0, alpha2old - alpha1old])
            high = min([self.c, self.c + alpha2old - alpha1old])
        else:
            low = max([0, alpha1old + alpha2old - self.c])
            high = min([self.c, alpha1old + alpha2old])

        return low, high

    def _kkt(self, i: int, ki: ndarray, yi: ndarray) -> bool:
        constraint = yi * (np.sum(self._alphas[i] * yi * ki) + self._b)
        alpha_i = self._alphas[i]
        if self.c:
            if (constraint > 1 and np.isclose(alpha_i, 0)) or \
                    (constraint < 1 and np.isclose(alpha_i, self.c)) or \
                    (np.isclose(constraint, 1) and 0 < alpha_i < self.c):
                return True
            else:
                return False
        else:
            if alpha_i >= 0 and constraint >= 0 and np.isclose(alpha_i * constraint, 0):
                return True
            else:
                return False
