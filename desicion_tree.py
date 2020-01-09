#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
决策树算法。
属性包含属性键、属性取值映射和属性值
"""

import math
from typing import Iterable, Union, Callable, Dict, Tuple, Optional, List

import numpy as np
from numpy import ndarray

import _inner_tools as _t
import model_selection as _ms
from base import ISuperviseLearner
from exception import DataNotMatchError, StateError
from mlfunc import ent, gini


def gain(x_mat: ndarray, y_row: ndarray, prop: int, prop_values: Iterable) -> float:
    """
    计算使用属性 prop 对样本集进行划分的信息增益，值越大表示使用属性 prop 进行划分
    所获得的纯度提升越大。此方法对可取值数目较多的属性有所偏好。

    :param x_mat: 特征向量组，行数 m 表示样本数，列数 n 表示特征数
    :param y_row: 输出向量。是一个只有一个维度的行向量，要和 x_mat 匹配
    :param prop: 进行划分的属性
    :param prop_values: 属性的取值
    :return: 信息增益
    """

    prop_x = x_mat[:, prop]
    prop_y = []
    for v in prop_values:
        prop_y.append(y_row[prop_x == v])

    ent_sum = 0
    m = y_row.shape[0]
    for y in prop_y:
        ent_sum = ent_sum + (len(y) / m) * ent(y)

    return ent(y_row) - ent_sum


def gain_ratio(x_mat: ndarray, y_row: ndarray, prop: int, prop_values: Iterable, gain_value: float = None) -> float:
    """
    计算使用属性 prop 对样本集进行划分的信息增益率，值越大表示使用属性 prop 进行划分
    所获得的纯度提升越大。此方法对可取值数目较少的属性有所偏好

    :param x_mat: 特征向量组，行数 m 表示样本数，列数 n 表示特征数
    :param y_row: 输出向量。是一个只有一个维度的行向量，要和 x_mat 匹配
    :param prop: 进行划分的属性
    :param prop_values: 属性的取值
    :param gain_value: 信息增益。给出该值是为了避免重复计算。
    :return: 信息增益率
    """

    prop_x = x_mat[:, prop]
    prop_y_num = []
    for v in prop_values:
        prop_y_num.append(len(y_row[prop_x == v]))

    m = y_row.shape[0]
    intrinsic_value = 0
    for num in prop_y_num:
        tmp = num / m
        intrinsic_value = intrinsic_value - tmp * (0 if math.isclose(tmp, 0) else math.log2(tmp))

    if gain_value is None:
        gain_value = gain(x_mat, y_row, prop, prop_values)

    return gain_value / intrinsic_value


def gain_prop_partition(x_mat: ndarray, y_row: ndarray, prop_dict: dict):
    """
    使用信息增益和信息增益率选择最优划分属性。

    :param x_mat: 特征向量组，行数 m 表示样本数，列数 n 表示特征数
    :param y_row: 输出向量，是一个 (m,) 的行向量，每一列代表 x_mat 中对应行的输出
    :param prop_dict: 属性-属性值的 dict。键为属性，值为属性值的列表
    :return: 最优划分属性
    """

    # 首先从划分属性中找出信息增益高于平均水平的属性，再从中选择增益率最高的

    average = 0
    gain_dict = {}
    for p, vs in prop_dict.items():
        g = gain(x_mat, y_row, p, vs)
        gain_dict[p] = g
        average = average + g
    average = average / len(gain_dict)
    gain_dict = {p: g for p, g in gain_dict.items() if g > average}

    best_p = None
    best_gr = 0
    for p, g in gain_dict.items():
        # 当两个属性 gr 相同时，进行不同的选择，可能会对之后的操作有很大的影响，会陷入到局部最优化中
        gr = gain_ratio(x_mat, y_row, p, prop_dict[p], g)
        if best_gr < gr:
            best_p = p
            best_gr = gr

    return best_p


def gini_index(x_mat: ndarray, y_row: ndarray, prop: int, prop_values: Iterable) -> float:
    """
    计算使用属性 prop 对样本集进行划分的基尼指数，值越小表示使用属性 prop 进行划分
    所获得的纯度提升越大。

    :param x_mat: 特征向量组，行数 m 表示样本数，列数 n 表示特征数
    :param y_row: 输出向量。是一个只有一个维度的行向量，要和 x_mat 匹配
    :param prop: 进行划分的属性
    :param prop_values: 属性的取值
    :return: 基尼指数
    """

    prop_x = x_mat[:, prop]
    prop_y = []
    for v in prop_values:
        prop_y.append(y_row[prop_x == v])

    gini_sum = 0
    m = y_row.shape[0]
    for y in prop_y:
        gini_sum = gini_sum + (len(y) / m) * gini(y)

    return gini_sum


def gini_prop_partition(x_mat: ndarray, y_row: ndarray, prop_dict: dict) -> float:
    """
    使用基尼值选择最优划分属性。

    :param x_mat: 特征向量组，行数 m 表示样本数，列数 n 表示特征数
    :param y_row: 输出向量，是一个 (m,) 的行向量，每一列代表 x_mat 中对应行的输出
    :param prop_dict: 属性-属性值的 dict。键为属性，值为属性值的列表
    :return: 最优划分属性
    """

    best_p = None
    best_g = 1
    for p, pv in prop_dict.items():
        g = gini_index(x_mat, y_row, p, pv)
        if best_g > g:
            best_p = p
            best_g = g

    return best_p


# noinspection PyAttributeOutsideInit
class _DecisionTreeNode:

    def __init__(self, *, prop: int = None, label: int = None, prop_values: Iterable = None):
        """
        创建一个决策树节点。可以给它分配属性、标签。

        :param prop: 这个节点用来进行判别的属性。叶节点这个值为 None
        :param label: 这个节点所属的类别。分支节点这个值为 None
        :param prop_values: 属性对应的值
        """

        self._prop = prop
        self._label = label
        self._vns = {}
        if prop is not None and prop_values is not None:
            for v in prop_values:
                self._vns[v] = None

    @property
    def label(self) -> int:
        return self._label

    @label.setter
    def label(self, value: int):
        self._label = value
        self._prop = None
        self._vns.clear()
        if hasattr(self, '_children'):
            del self._children
        if hasattr(self, '_values'):
            del self._values

    @property
    def prop(self) -> int:
        return self._prop

    def is_leaf(self) -> bool:
        return self._prop is None and self._label is not None

    def set_prop(self, prop: int, prop_values: Iterable):
        """
        为这个结点分配一个属性，此结点为分支节点。

        :param prop: 这个节点用来进行判别的属性。
        :param prop_values: 属性对应的值
        """

        self._prop = prop
        for v in prop_values:
            self._vns[v] = None
        self._label = None

    def put_child(self, prop_value: int, child):
        if self.is_leaf():
            raise ValueError('leaf node cannot put child node')
        if prop_value not in self._vns:
            raise ValueError('property value does not exist')

        self._vns[prop_value] = child

    def get_child(self, prop_value: int):
        if self.is_leaf():
            raise ValueError('leaf node cannot get child node')
        if prop_value not in self._vns:
            raise ValueError('property value does not exist')

        return self._vns[prop_value]

    def children_num(self) -> int:
        if self.is_leaf():
            raise ValueError('leaf node cannot put child node')

        return len(self._vns)

    def children(self) -> Tuple:
        if self.is_leaf():
            raise ValueError('leaf node cannot put child node')
        if not hasattr(self, '_children'):
            self._children = tuple((child for child in self._vns.values()))

        return self._children

    def values(self) -> Tuple[int]:
        if self.is_leaf():
            raise ValueError('leaf node cannot put child node')
        if not hasattr(self, '_values'):
            self._values = tuple((v for v in self._vns.keys()))

        return self._values

    def sl_prop(self, save: bool = True, *, delete: bool = False):
        if delete:
            if hasattr(self, '_tmp_prop'):
                del self._tmp_prop
                del self._tmp_vns
            return

        if save:
            self._tmp_prop = self._prop
            self._tmp_vns = self._vns.copy()
        else:
            self._prop = self._tmp_prop
            self._vns = self._tmp_vns


class DecisionTree(ISuperviseLearner):
    """
    决策树。
    """

    def __init__(self, *, prop_value_map: Union[Tuple[Union[Iterable[int], None]], List[Union[Iterable[int], None]],
                                                Dict[int, Union[Iterable[int], None]]],
                 prop_partition: Union[str, Callable[[ndarray, ndarray, Dict], int]] = 'gain',
                 x_cv: ndarray = None, y_cv: ndarray = None, pruning: str = None):
        """
        初始化决策树。使用样本集、标签、属性和最有属性划分函数进行初始化。

        :param prop_value_map: 属性-属性值的映射。如果它是 tuple 或 list，那么下标为属性，值为属性对应的取值；如果它是 dict，
                               那么键为属性，值为属性对应的取值。其中，属性就是数据集中特征向量的下标。如果属性是离散属性，那么
                               属性值也就是所有的离散值；如果属性是连续属性，那么令属性值为 None 表示这是一个连续属性
        :param prop_partition: 最有属性划分方法。使用字符串表示使用内部划分方法，字符串取值有'gain'、'gini'等；也可以提供自己的
        划分方法，划分方法必须接受 x_mat,y_row,prop_dict 参数，输出最优属性
        :param x_cv: 验证输入集，用于对决策树的剪枝，为 None 表示不剪枝
        :param y_cv: 验证输出集，用于对决策树的剪枝。为 None 表示不剪枝
        :param pruning: 剪枝方式，'post' 后剪枝，'pre' 预剪枝，为 None 表示不剪枝
        """

        self.pvs = prop_value_map
        self.prop_partition = prop_partition
        self.pruning = pruning
        self.x_cv = x_cv
        self.y_cv = y_cv

        self.__check_params()

        self._size = 0
        self._root = None

    def __sizeof__(self):
        return self._size

    @property
    def pvs(self) -> Dict[int, Union[Iterable[int], None]]:
        return self._pvs

    @pvs.setter
    def pvs(self, value: Union[Tuple[Union[Iterable[int], None]], List[Union[Iterable[int], None]],
                               Dict[int, Union[Iterable[int], None]]]):
        if isinstance(value, dict):
            self._pvs = value
        else:
            self._pvs = {}
            for p, v in enumerate(value):
                self._pvs[p] = v

    @property
    def prop_partition(self) -> Callable[[ndarray, ndarray, Dict], int]:
        return self._prop_partition

    @prop_partition.setter
    def prop_partition(self, value: Union[str, Callable[[ndarray, ndarray, Dict], int]]):
        if isinstance(value, str):
            if value == 'gain':
                self._prop_partition = gain_prop_partition
            elif value == 'gini':
                self._prop_partition = gini_prop_partition
            else:
                raise ValueError('prop_partition must be one of "gain" "gini", or be a function')
        else:
            self._prop_partition = value

    @property
    def pruning(self) -> Optional[str]:
        return self._pruning

    @pruning.setter
    def pruning(self, value: Optional[str]):
        if value is not None and value != 'post' and value != 'pre':
            raise ValueError('pruning must be "post" or "pre"')

        self._pruning = value

    @property
    def x_cv(self) -> Optional[ndarray]:
        return self._x_cv

    @x_cv.setter
    def x_cv(self, value: Optional[ndarray]):
        if value is not None:
            self._x_cv = _t.r2m(value)
        else:
            self._x_cv = None

    @property
    def y_cv(self) -> Optional[ndarray]:
        return self._y_cv

    @y_cv.setter
    def y_cv(self, value: Optional[ndarray]):
        if value is not None:
            self._y_cv = _t.c2r(value)
        else:
            self._y_cv = None

    @property
    def size(self) -> int:
        return self._size

    # noinspection PyAttributeOutsideInit
    def train(self, x_mat: ndarray, y_row: ndarray):
        """
        使用给定的数据对 LDA 学习算法进行训练。

        :param x_mat: 特征向量组，行数 m 表示样本数，列数 n 表示特征数
        :param y_row: 输出行向量，每一个值代表 x_mat 中对应行的输出
        """

        self.__check_params()
        x_mat, y_row = _t.match_x_y(x_mat, y_row, add_ones=False)
        if x_mat.shape[1] != len(self.pvs):
            raise DataNotMatchError('feature quantity mismatch')

        is_pruning = self.x_cv is not None and self.y_cv is not None
        if is_pruning:
            self._node0 = None
            self._pruning_accuracy = 0
            if self.pruning == 'pre':
                # 预剪枝先考虑只有一个叶结点的情况
                self._root = _DecisionTreeNode()
                self._root.label = self.__find_most_label(y_row)
                self.__pruning_test(self._root)
                # 递归地进行预剪枝
                self._root = self.__generate(x_mat, y_row, self.pvs.copy(), cur_layer=0, max_layer=1)
            else:
                # 后剪枝需要先生成整棵树，然后从底层的非叶节点开始进行剪枝，比较是否得到了性能提升
                self._stack = []
                self._root = self.__generate(x_mat, y_row.ravel(), self.pvs.copy())
                self.__pruning_test(self._root)
                while len(self._stack) > 0:
                    node, label = self._stack.pop(-1)
                    children_num = node.children_num()
                    node.sl_prop()
                    node.label = label
                    # 如果剪枝后性能得到提升，就剪枝，否则就保留
                    if self.__pruning_test(self._root):
                        self._size = self._size - children_num
                        node.sl_prop(delete=True)
                    else:
                        node.sl_prop(False)
        else:
            self._root = self.__generate(x_mat, y_row.ravel(), self.pvs.copy())

    def predict(self, x_mat: ndarray) -> Union[ndarray, int]:
        """
        对新的数据进行预测。当 x_mat 中只有一个样本时返回一个数字，否则返回一个行向量。

        :param x_mat: 特征向量组，行数 m 表示样本数，列数 n 表示特征数
        :return: 预测值或向量
        """

        if self._root is None:
            raise StateError('not trained yet')

        x_mat = _t.r2m(x_mat)
        if x_mat.shape[1] != len(self.pvs):
            raise DataNotMatchError('feature quantity mismatch')

        return self.__predict(self._root, x_mat)

    def cost(self, x_mat: ndarray, y_row: ndarray) -> float:
        """
        计算在 x_mat 和 y_row 上的代价。

        :param x_mat: 特征向量组，行数 m 表示样本数，列数 n 表示特征数
        :param y_row: 输出行向量，每一个值代表 x_mat 中对应行的输出
        :return: 代价值。
        """

        if self._root is None:
            raise StateError('not trained yet')

        x_mat, y_row = _t.match_x_y(x_mat, y_row, add_ones=False)
        if x_mat.shape[1] != len(self.pvs):
            raise DataNotMatchError('feature quantity mismatch')

        # TODO: 暂时先用错误率作为代价值，以后想想有什么更好的方法
        return 1 - _ms.accuracy(self.__predict(self._root, x_mat), y_row)

    def __check_params(self):
        pvs = self.pvs
        x_cv = self.x_cv
        y_cv = self.y_cv
        pruning = self.pruning

        if x_cv is not None and x_cv.shape[1] != len(pvs):
            raise DataNotMatchError('feature quantity mismatch')
        if (x_cv is not None and y_cv is None) or (x_cv is None and y_cv is not None):
            raise ValueError('x_val and y_val must all be none or all exist')
        if (x_cv is not None and y_cv is not None) and (x_cv.shape[0] != y_cv.shape[0]):
            raise DataNotMatchError('number of samples does not match')
        if (x_cv is not None and y_cv is not None) and pruning is None:
            raise ValueError('pruning must be "post" or "pre"')

    def __generate(self, x_mat: ndarray, y_row: ndarray, prop_dict: dict, *,
                   cur_layer: int = None, max_layer: int = None,
                   add_in_parent: Callable[[_DecisionTreeNode], None] = None) -> _DecisionTreeNode:
        """
        递归地生成决策树。

        :param x_mat: 特征向量组，行数 m 表示样本数，列数 n 表示特征数
        :param y_row: 输出向量，是一个 (m,) 的行向量，每一列代表 x_mat 中对应行的输出
        :param prop_dict: 属性-属性值的 dict。键为属性，值为属性值的列表
        :param cur_layer: 当前层数，层数从 0 开始，用在预剪枝中，为 None 表示不剪枝
        :param max_layer: 剪枝的最大层数，用在预剪枝中，为 None 表示不剪枝
        :param add_in_parent: 用来将结点加入到它的父结点中的函数，用在预剪枝中
        :return: 一个 _DeTreeNode 节点
        """

        # 预剪枝先在底层进行剪枝，再回到上一层进行性能评估，如果性能得到提升，那么上一层的展开就是有效的，就继续继续下去；
        # 反之，则上一层就不展开。
        # 我们在不断向下探索的过程中，需要注意保存已经展开的结点，不能重复运算

        m = y_row.shape[0]
        node = _DecisionTreeNode()
        self._size = self._size + 1

        # 如果数据集中样本全属于同一类别 C，将 node 标记为 C 类叶结点
        if (y_row == y_row[0]).all():
            node.label = y_row[0]
            return node

        # 如果属性集为空，或样本集中样本在所有属性上取值相同，无法划分，将 node 标记为样本数最多的类的叶结点
        prop_check = x_mat[:, list(prop_dict.keys())]
        if len(prop_dict) == 0 or ((np.sum(prop_check, axis=0) / m) == prop_check[0]).all():
            node.label = self.__find_most_label(y_row)
            return node

        # 进行预剪枝操作，将非叶节点变成叶节点，从而防止展开
        if cur_layer is not None:
            # 保存第一个结点
            if self._node0 is None and cur_layer == 0:
                self._node0 = node

            # 如果当前层等于最大层，对这一层进行剪枝
            if cur_layer == max_layer:
                node.label = self.__find_most_label(y_row)
                return node

        # 选取最优属性
        best_p = self.prop_partition(x_mat, y_row, prop_dict)
        node.set_prop(best_p, prop_dict[best_p])
        # 后剪枝保存所有非叶结点和它们对应的标签数据
        if self.pruning == 'post':
            self._stack.append((node, self.__find_most_label(y_row)))
        # 预剪枝之前，保存需要继续展开的子结点信息
        spread_vs = None
        if cur_layer is not None and cur_layer == max_layer - 1:
            spread_vs = []
        for v in prop_dict[best_p]:
            # 选取样本集在 best_p 上取值为 v 的样本子集
            x_v = x_mat[x_mat[:, best_p] == v, :]
            if x_v.shape[0] == 0:
                # x_v 为空将 child 标记为样本数最多的类的叶结点
                self._size = self._size + 1
                node.put_child(v, _DecisionTreeNode(label=self.__find_most_label(y_row)))
            else:
                if spread_vs is not None:
                    spread_vs.append(v)
                # 递归地生成其他结点
                vs = prop_dict.pop(best_p)
                node.put_child(v, self.__generate(x_v, y_row[x_mat[:, best_p] == v], prop_dict,
                                                  cur_layer=cur_layer + 1 if cur_layer is not None else None,
                                                  max_layer=max_layer,
                                                  add_in_parent=(lambda child: node.put_child(v, child))
                                                  if cur_layer is not None else None))
                if add_in_parent:
                    add_in_parent(node)
                prop_dict[best_p] = vs

        # 如果本层结点是剪枝结点的父结点的话，就进行性能判断
        if cur_layer is not None and cur_layer == max_layer - 1:
            self._size = self._size - node.children_num()
            if self.__pruning_test(self._node0):
                # 性能得到提升的话，就继续展开
                # 如果有子结点可以展开的话
                if len(spread_vs) > 0:
                    x_p = x_mat[:, best_p]
                    for v in spread_vs:
                        vs = prop_dict.pop(best_p)
                        node.put_child(v, self.__generate(x_mat[x_p == v, :], y_row[x_p == v], prop_dict,
                                                          cur_layer=cur_layer + 1, max_layer=max_layer + 1,
                                                          add_in_parent=(lambda child: node.put_child(v, child))))
                        if add_in_parent:
                            add_in_parent(node)
                        prop_dict[best_p] = vs
            else:
                # 否则进行剪枝
                node.label = self.__find_most_label(y_row)

        return node

    @staticmethod
    def __predict(node: _DecisionTreeNode, x_mat: ndarray):
        """
        从 node 开始，对新的数据进行预测。当 x_mat 中只有一个样本时返回一个数字，否则返回一个行向量。

        :param node: 决策树结点
        :param x_mat: 一个样本
        :return: 此样本对应的类别
        """

        r = np.empty((x_mat.shape[0],))
        for i, x in enumerate(x_mat):
            n = node
            while not n.is_leaf():
                n = n.get_child(x[n.prop])
            r[i] = n.label

        return _t.ret(r)

    @staticmethod
    def __find_most_label(y_row: ndarray) -> int:
        """
        找到 y_row 中最多的类别。

        :param y_row: 输出向量
        :return: 类别
        """

        label_counts = {}
        for y in y_row:
            if y in label_counts:
                label_counts[y] = label_counts[y] + 1
            else:
                label_counts[y] = 1

        max_y = 0
        max_count = 0
        for y, count in label_counts.items():
            if max_count < count:
                max_y = y
                max_count = count

        return max_y

        # return sorted([(np.sum(y_row == label), label) for label in set(y_row)])[-1][1]

    def __pruning_test(self, node: _DecisionTreeNode) -> bool:
        """
        测试剪枝后性能有没有提高。

        :param node 开始搜索的结点
        :return: True 表示提高了，False 表示没有
        """

        accuracy = np.mean(self.__predict(node, self.x_cv) == self.y_cv) * 100
        if self._pruning_accuracy < accuracy:
            self._pruning_accuracy = accuracy
            return True

        return False
