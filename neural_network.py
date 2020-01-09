#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
神经网络相关算法
"""

import math
from queue import Queue
from typing import Union, Tuple, List, Optional, Callable, Iterable, Dict

import numpy as np
import scipy.optimize as op
from numpy import ndarray

import _inner_tools as _t
import mlfunc as _mf
from base import IProbabilityLearner
from exception import DataNotMatchError, StateError


class FNN(IProbabilityLearner):
    """
    前馈（Feedforward）神经网络。
    """

    def __init__(self, *, labels: Union[None, Tuple[int], List[int], ndarray] = None,
                 layer_nodes: Union[Tuple[int], List[int], ndarray],
                 lamb: float = 0, max_iter: int = 100, threshold: float = 0.5):
        """
        初始化前馈神经网络。

        :param labels: 类别数组，包含所有类别的值，为 None 表示训练的是二分类问题，且类别标记为 1, 0。
                       在二分类问题中，默认排在前面的类别是正类
        :param layer_nodes: 神经网络每层神经元数，包括输入层、隐藏层和输出层。
        :param lamb: 正则化参数，默认为 0
        :param max_iter: 训练的最大迭代次数，默认为 100
        :param threshold: 阈值，默认为 0.5，用在二分类问题中
        """

        self.labels = labels
        self.layer_nodes = layer_nodes
        self.lamb = lamb
        self.max_iter = max_iter
        self.threshold = threshold

        self.__check_params()

        self._thetas = None

    @property
    def lamb(self) -> Union[int, float]:
        return self._lamb

    @lamb.setter
    def lamb(self, value: Union[int, float]):
        if value < 0:
            raise ValueError('lamb must be greater than or equal to 0')
        self._lamb = value

    @property
    def layer_nodes(self) -> ndarray:
        return self._layer_nodes

    @layer_nodes.setter
    def layer_nodes(self, value: Union[Tuple[int], List[int], ndarray]):
        if len(value) < 3:
            raise ValueError('layer_nodes must have at least three levels')

        if not isinstance(value, ndarray):
            self._layer_nodes = np.asarray(value)
        else:
            self._layer_nodes = value

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
    def thetas(self) -> Optional[List[ndarray]]:
        return self._thetas

    def train(self, x_mat: ndarray, y_row: ndarray) -> List[ndarray]:
        """
        训练前馈神经网络，既可以训练二分类问题，也可以训练多类分类问题。

        :param x_mat: 特征向量组，行数 m 表示样本数，列数 n 表示特征数
        :param y_row: 输出行向量，每一个值代表 x_mat 中对应行的输出
        :return: 每层的权值（输出层没有）。
        """

        self.__check_params()
        x_mat, y_row = _t.match_x_y(x_mat, y_row)
        if x_mat.shape[1] - 1 != self.layer_nodes[0]:
            raise DataNotMatchError('feature number and input layer node number mismatch')
        y_row = _t.convert_y(self.labels, y_row)

        init_theta = np.empty((0,))
        for i in range(len(self.layer_nodes) - 1):
            theta = self.__rand_thetas(self.layer_nodes[i], self.layer_nodes[i + 1])
            init_theta = np.hstack((init_theta, theta.ravel()))

        self._thetas = self.__extract_thetas(op.fmin_cg(
            f=lambda t, x, y: self.__cost(self.__extract_thetas(t), x, y),
            x0=init_theta, args=(x_mat, y_row), maxiter=self.max_iter,
            fprime=lambda t, x, y: self.__gradient(self.__extract_thetas(t), x, y)))

        return self._thetas

    def predict(self, x_mat: ndarray) -> Union[ndarray, int]:
        """
        返回预测值，是对应于 x_mat 的标记。

        :param x_mat: 特征向量组，行数 m 表示样本数，列数 n 表示特征数
        :return: 预测标记
        """

        if self._thetas is None:
            raise StateError('not trained yet')

        x_mat = _t.addones(x_mat)
        if x_mat.shape[1] - 1 != self.layer_nodes[0]:
            raise DataNotMatchError('feature number and input layer node number mismatch')

        a = self.__feedforward(self._thetas, x_mat)[-1]
        if len(self.labels) == 2:
            return _t.ret(_t.convert_y(self.labels, _t.c2r(a >= self.threshold), to=False))
        else:
            return _t.ret(self.labels[np.argmax(a, axis=1)])

    def cost(self, x_mat: ndarray, y_row: ndarray) -> float:
        """
        计算在 x_mat 和 y_row 上的代价。

        :param x_mat: 特征向量组，行数 m 表示样本数，列数 n 表示特征数
        :param y_row: 输出行向量，每一个值代表 x_mat 中对应行的输出
        :return: 代价值。
        """

        if self._thetas is None:
            raise StateError('not trained yet')

        x_mat, y_row = _t.match_x_y(x_mat, y_row)
        if x_mat.shape[1] - 1 != self.layer_nodes[0]:
            raise DataNotMatchError('feature number and input layer node number mismatch')

        return self.__cost(self._thetas, x_mat, y_row)

    def probability(self, x_mat: ndarray) -> Union[ndarray, float]:
        """
        返回对应于 x_mat 的预测概率。如果是二分类问题，那么返回一个行向量；如果是多分类问题，返回一个
        m*num_labels 的矩阵，其中每一行表示样本在每个类上的概率。

        :param x_mat: 特征向量组，行数 m 表示样本数，列数 n 表示特征数
        :return: 预测概率。
        """

        if self._thetas is None:
            raise StateError('not trained yet')

        x_mat = _t.addones(x_mat)
        if x_mat.shape[1] - 1 != self.layer_nodes[0]:
            raise DataNotMatchError('feature number and input layer node number mismatch')

        return _t.ret(_t.c2r(self.__feedforward(self._thetas, x_mat)[-1]))

    def __cost(self, thetas: Union[List[ndarray], Tuple[ndarray]], x_mat: ndarray, y_row: ndarray) -> float:
        """
        计算代价值。

        :param thetas: 所有层的参数
        :param x_mat: 特征向量组，行数 m 表示样本数，列数 n 表示特征数
        :param y_row: 输出行向量，每一个值代表 x_mat 中对应行的输出
        :return: 代价值
        """

        an = self.__feedforward(thetas, x_mat)[-1]
        m = x_mat.shape[0]

        if len(self.labels) > 2:
            y = np.zeros((m, self.labels.shape[0]))
            for i in range(m):
                y[i, np.argwhere(self.labels == y_row[i]).ravel()[0]] = 1
            # 注意，多分类问题中这里是点乘
            temp = np.sum((-y * np.log(an) - (1 - y) * np.log(1 - an)).ravel()) / m
        else:
            temp = np.sum((-y_row @ np.log(an) - (1 - y_row) @ np.log(1 - an)).ravel()) / m

        cost = _t.ret(temp)
        form = 0
        if self.lamb != 0:
            for theta in thetas:
                form += np.sum(theta[:, 1:].ravel() ** 2)

        return _t.ret(cost + self.lamb * form / (2 * m))

    def __gradient(self, thetas: Union[List[ndarray], Tuple[ndarray]], x_mat: ndarray, y_row: ndarray) -> ndarray:
        """
        计算梯度。

        :param thetas: 所有层的参数
        :param x_mat: 特征向量组，行数 m 表示样本数，列数 n 表示特征数
        :param y_row: 输出行向量，每一个值代表 x_mat 中对应行的输出
        :return: 梯度，是一个行向量
        """

        feeds = self.__feedforward(thetas, x_mat)
        m = x_mat.shape[0]
        # deltas 存储了 delta_n, delta_(n-1)..., delta_2
        deltas = []
        # grads 存储了 thetas 对应的梯度
        grads = []

        if self.labels is not None:
            # 在多参数的情况下将 y_col 展开为 m*num_labels 的矩阵，每一行上对应类别处为 1
            y = np.zeros((m, self.labels.shape[0]))
            for i in range(m):
                y[i, np.argwhere(self.labels == y_row[i]).ravel()[0]] = 1
        else:
            y = y_row

        # ans 存储 an...a2，zns 存储 zn...z2
        ans = []
        zns = []
        for i, f in enumerate(feeds):
            if i % 2 == 1:
                ans.append(f)
            else:
                zns.append(f)
        # 倒过来方便计算，因为后向传播要从后往前计算
        ans.reverse(), zns.reverse(), thetas.reverse()
        # 计算 delta_n
        deltas.append(ans[0] - y)
        # 删除 an 和 zn
        ans.pop(0), zns.pop(0)
        # 计算 delta_(n-1)...delta_2
        for z, theta in zip(zns, thetas):
            deltas.append((deltas[-1] @ theta)[:, 1:] * _mf.sigmoid_gradient(z))
        # 将 x_mat 作为 a_1 加入进来
        ans.append(x_mat)
        # 计算梯度
        for d, a in zip(deltas, ans):
            grads.append((d.T @ a) / m)
        # 计算正则项
        if self.lamb != 0:
            for i in range(len(grads)):
                grads[i] = grads[i] + self.lamb * np.hstack((np.zeros((thetas[i].shape[0], 1)), thetas[i][:, 1:])) / m
        # 结果要从 1...n，所以需要倒过来
        grads.reverse()

        grad = np.empty((0,))
        for g in grads:
            grad = np.hstack((grad, g.ravel()))

        return grad

    @staticmethod
    def __feedforward(thetas: Union[List[ndarray], Tuple[ndarray]], x_mat: ndarray) -> list:
        """
        正向传播，计算分类问题的预测概率。

        :param x_mat: 特征向量组，行数 m 表示样本数，列数 n 表示特征数
        :param thetas: 参数向量组。每一个 theta 表示对应层的参数
        :return: 一个 list，存有每一层计算后的 ai 和 zi（除了 an，其他 ai 都加上了一列偏置列）
        """

        r = []
        a = x_mat
        for i, theta in enumerate(thetas):
            z = a @ theta.T
            r.append(z)
            a = _mf.sigmoid(z)
            if i != len(thetas) - 1:
                a = np.hstack((np.ones((a.shape[0], 1)), a))
            r.append(a)

        return r

    def __extract_thetas(self, theta_row: ndarray) -> List[ndarray]:
        thetas = []
        offset = 0
        for i in range(len(self.layer_nodes) - 1):
            theta_nums = (self.layer_nodes[i] + 1) * self.layer_nodes[i + 1]
            thetas.append(theta_row[offset:theta_nums + offset]
                          .reshape((self.layer_nodes[i + 1], self.layer_nodes[i] + 1)))
            offset = offset + theta_nums

        return thetas

    @staticmethod
    def __rand_thetas(in_size: int, out_size: int) -> ndarray:
        """
        根据输入层和输出层的大小生成随机的参数。
        需要注意的是输入层的大小不包含偏置项。

        :param in_size: 输入层大小
        :param out_size: 输出层大小
        :return: 参数矩阵
        """

        epsilon_init = math.sqrt(6) / math.sqrt(out_size + in_size)
        return np.random.random_sample((out_size, 1 + in_size)) * 2 * epsilon_init - epsilon_init

    def __check_params(self):
        if (len(self.labels) == 2 and self.layer_nodes[-1] != 1) or \
                (len(self.labels) != self.layer_nodes[-1]):
            raise DataNotMatchError('number of labels does not match number of output layer nodes')


"""
多层感知机，等价于前馈神经网络
"""
MLP = FNN

# 可以表示函数输出输入值或梯度
Value = Union[ndarray, int, float]
# 一元函数
Unary_fn = Callable[[Value], Value]
# 二元函数
Binary_fn = Callable[[Value, Value], Value]
# 计算图操作函数
Gfn = Union[Unary_fn, Binary_fn]
# 反向传播操作函数，返回梯度
Bprop = Union[Callable[[Value, Value, Value], Value], Callable[[Value, Value, Value, Value], Value]]


class _Operation:
    """
    用来表示计算图中操作的对象。每个操作包含一个操作实现的计算函数和一个与之对应的
    反向传播函数（也就是梯度函数）。
    """

    def __init__(self, fn: Gfn, bp: Bprop):
        """
        初始化操作对象。

        :param fn: 操作实现的数学函数。它接受一个或两个变量，输出计算结果
        :param bp: 与 fn 对应的反向传播操作，用来计算梯度。它接受提供给操作的一组输入变量（计算图中的父结点）、
                   需要计算梯度的变量（此操作对应的变量）和对于输出变量（计算图中的子结点）的梯度。最终输出梯度
        """

        self.fn = fn
        self.bp = bp


_KV = 'value'
_KP = 'parent'
_KC = 'child'
_KO = 'operation'


# noinspection PyTypeChecker
class _ComputationGraph:
    """
    计算图，一个表示某种计算形式的有向无环图。每个结点是一个变量，每条边是一个操作。图中每个节点至多有两个父结点（输入），
    子结点（输出）可以有零个、一个或多个。我们可以从图中获取节点的父结点、子结点和产生结点的操作。
    计算图可以有多个开始变量，开始变量没有父结点；只有唯一的结束变量，也就是计算的结果，结束变量没有子结点。
    默认约定，每个节点的父结点是一个序列或单个值，它的排列顺序和此节点对应操作的参数顺序一致。
    """

    def __init__(self, variable_num: int, operations: Dict[int, _Operation], targets: Union[Iterable[int], ndarray]):
        """
        初始化计算图。

        :param variable_num: 图中变量数
        :param operations: 用来产生每个变量的操作集合。其中键是变量的标号，值是操作
        :param targets: 需要计算梯度的目标变量集
        """

        self.targets = targets
        # 表示计算图，键是结点的标号，是一个整数值（从0开始）；值是一个字典，这个字典包含父结点、子结点、值和操作的键值对
        self._graph = {}
        for v in range(variable_num):
            self._graph[v] = {_KV: None, _KP: None, _KC: None, _KO: operations.get(v, None)}

        self._starts = None
        self._end = None

    def __sizeof__(self):
        return len(self._graph)

    def __getitem__(self, item):
        return self.get_value(item)

    def __setitem__(self, key, value):
        self.put_value(key, value)

    @property
    def targets(self) -> ndarray:
        return self._targets

    @targets.setter
    def targets(self, value: Union[Iterable[int], ndarray]):
        if isinstance(value, ndarray):
            self._targets = value
        else:
            self._targets = np.asarray(value)

    def put_value(self, v: int, value: Value):
        self._graph[v][_KV] = value

    def get_value(self, v: int):
        return self._graph[v][_KV]

    def put_parent(self, v: int, p: Union[Iterable[int], int]):
        if isinstance(p, (ndarray, int)):
            self._graph[v][_KP] = p
        else:
            self._graph[v][_KP] = np.asarray(p)

    def get_parent(self, v: int) -> Union[ndarray, int, None]:
        return self._graph[v][_KP]

    def put_child(self, v: int, c: Union[Iterable[int], int]):
        if isinstance(c, ndarray):
            self._graph[v][_KC] = c
        else:
            self._graph[v][_KC] = np.asarray(c)

    def get_child(self, v: int) -> Union[ndarray, None]:
        return self._graph[v][_KC]

    def put_operation(self, v: int, ope: _Operation):
        self._graph[v][_KO] = ope

    def get_operation(self, v: int) -> _Operation:
        return self._graph[v][_KO]

    def find_endpoint(self):
        """
        找出计算图的起始变量集和结束变量
        """

        self._starts = []
        for v, g in self._graph.items():
            if g[_KP] is None:
                self._starts.append(v)
            if g[_KC] is None:
                self._end = v

    # noinspection PyCallingNonCallable
    def fprop(self, endpoint: bool = False):
        """
        正向传播。从开始变量（没有父结点的变量）出发，进行一次计算，直到结束变量（没有子结点的变量）。
        需要注意的是，如果开始变量或操作为 None，将会出错。

        :param endpoint: 是否重新计算开始变量和结束变量。默认为 False
        :return 最终的计算结果，也就是结束变量的值
        """

        if endpoint or self._starts is None:
            self.find_endpoint()

        # 宽度优先遍历
        q = Queue()
        mark = np.zeros(len(self._graph), dtype=bool)
        for v in self._starts:
            q.put(v)
            mark[v] = True

        while not q.empty():
            v = q.get()
            child = self._graph[v][_KC]
            if child is not None:
                for c in child:
                    # 避免重复计算
                    if not mark[c]:
                        p = self._graph[c][_KP]
                        if isinstance(p, ndarray):
                            # 如果计算步骤还未到这里，则暂时跳过
                            if not mark[p[p != v]]:
                                q.put(v)
                                break
                            self._graph[c][_KV] = self._graph[c][_KO](self._graph[p[0]][_KV],
                                                                      self._graph[p[1]][_KV])
                        else:
                            self._graph[c][_KV] = self._graph[c][_KO](self._graph[p][_KV])
                        q.put(c)
                        mark[c] = True

        return self._graph[self._end][_KV]

    def bprop(self, *, endpoint: bool = False, fprop: bool = True) -> List:
        """
        进行反向传播计算结束变量对所有目标变量的梯度。

        :param endpoint: 是否重新计算开始变量和结束变量。默认为 False
        :param fprop: 在进行反向传播前是否先进行正向传播，默认为 True
        :return: 目标变量的梯度列表
        """

        grad_table = [None for i in range(self._graph)]
        if fprop:
            self.fprop(endpoint=endpoint)
        grad_table[self._end] = 1

        for v in self.targets:
            self._build_grad(v, grad_table)

        return [grad for v, grad in grad_table if v in self.targets]

    def _build_grad(self, v: int, grad_table: List) -> ndarray:
        """
        递归地计算梯度。

        :param v: 需要被计算梯度的变量
        :param grad_table: 梯度表
        :return: v 的梯度
        """

        if grad_table[v] is not None:
            return grad_table[v]

        grad = 0
        for c in self._graph[v][_KC]:
            op = self._graph[c][_KO]
            d = self._build_grad(c, grad_table)
            p = self._graph[c][_KP]
            if isinstance(p, ndarray):
                grad = grad + op.bp(self._graph[p[0]][_KV], self._graph[p[1]][_KV], self._graph[v][_KV], d)
            else:
                grad = grad + op.bp(self._graph[p][_KV], self._graph[v][_KV], d)
        grad_table[v] = grad

        return grad
