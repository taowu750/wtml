#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
这个模块包含了关于模型评估和选择参数等的算法。

留出法：将数据集分成互斥的两部分，分别作为训练集、测试集
交叉验证法：将数据集分成 k 个大小相似的互斥子集，k-1个作为训练集，剩下一个作为测试集，进行 k 次
自助法：对数据集进行 m 次随机采样，得到一个样本集，使用这个样本集进行训练，使用未被选到的样本作为测试集

调参还需要验证集
"""

from collections import OrderedDict
from typing import Union, Tuple, List, Dict, Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray

import _inner_tools as _t
from base import ISuperviseLearner, IConfidenceLearner, IProbabilityLearner
from exception import DataNotMatchError, StateError


class Two2MultiClassLearner(IProbabilityLearner):
    """
    将多个二分类学习器组合训练出一个多分类学习器。
    """

    def __init__(self, *,
                 learner: Callable[[int, int], Union[ISuperviseLearner, IConfidenceLearner, IProbabilityLearner]],
                 labels: Union[ndarray, List[int], Tuple[int]], strategy: str = 'ovr',
                 positive_label: Optional[int] = None, negative_label: Optional[int] = None):
        """
        进行初始化。

        :param learner: 用来产生学习器的函数。
        :param labels: 类别数组，包含所有类别的值。至少要包含 3 个类别。
        :param strategy: 训练策略。有 ovo（一对一）和 ovr（一对其余）两种。ovo 将 N 个类别两两配对，从而产生 N(N-1)/2 个分类
                         任务，最终结果通过投票产生。ovr 则是每次将一个类的样例作为正例，所有其他类作为反例来训练 N 个分类器，
                         最终结果选用具有最大置信度的那个。ovo 的二分类器要求是 ISuperviseLearner 的子类，而 ovr 的二分类器
                         要求是 IConfidenceLearner 的子类。默认策略为 ovr。一般情况下 ovr 训练速度优于 ovo，而当类别很多时，
                         ovo 训练速度要优于 ovr。
        :param positive_label: 输入给二分类器的正例标签。如果为 None 表示将直接使用原有类别作为正例。
        :param negative_label: 输入给二分类器的反例标签。如果为 None 表示将直接使用原有类别作为反例。
        """

        if positive_label is not None and negative_label is not None and positive_label == negative_label:
            raise ValueError('positive_label and negative_label cannot be the same')

        self.learner = learner
        self.labels = labels
        self.strategy = strategy
        self.positive_label = positive_label
        self.negative_label = negative_label

        self._learners = OrderedDict()
        self._trained = False

    @property
    def labels(self) -> ndarray:
        return self._labels

    @labels.setter
    def labels(self, value: Union[ndarray, Tuple[int], List[int]]):
        if len(value) < 3:
            raise ValueError('labels contains at least three classes')

        if not isinstance(value, ndarray):
            self._labels = np.asarray(value)
        else:
            self._labels = value

    @property
    def strategy(self) -> str:
        return self._strategy

    @strategy.setter
    def strategy(self, value: str):
        if value != 'ovo' and value != 'ovr':
            raise ValueError('strategy must be "ovo" or "ovr"')

        self._strategy = value

    def train(self, x_mat: ndarray, y_row: ndarray):
        """
        进行训练。

        :param x_mat: 特征向量组，行数 m 表示样本数，列数 n 表示特征数
        :param y_row: 输出行向量，每一个值代表 x_mat 中对应行的输出
        """

        if self.positive_label is not None and self.negative_label is not None and \
                self.positive_label == self.negative_label:
            raise ValueError('positive_label and negative_label cannot be the same')

        x_mat, y_row = _t.match_x_y(x_mat, y_row, add_ones=False)
        self._trained = True
        self._learners.clear()

        if self.strategy == 'ovr':
            for label in self._labels:
                positive_label = label if self.positive_label is None else self.positive_label
                negative_label = label + 1 if self.negative_label is None else self.negative_label
                learner = self.learner(positive_label, negative_label)
                self._learners[label] = learner

                y = y_row.copy()
                # 记录更改之前的坐标，防止 positive_label 和 negative 相等或 negative_label 和 positive 相等这种情况
                positive_idx = y == label
                negative_idx = y != label
                if self.positive_label is not None:
                    y[positive_idx] = self.positive_label
                y[negative_idx] = negative_label
                learner.train(x_mat, y)
        else:
            for positive in self.labels:
                for negative in self.labels:
                    if positive == negative:
                        continue
                    positive_label = positive if self.positive_label is None else self.positive_label
                    negative_label = negative if self.negative_label is None else self.negative_label
                    learner = self.learner(positive_label, negative_label)
                    self._learners[(positive, negative)] = learner

                    y = y_row[(y_row == positive) | (y_row == negative)]
                    x = x_mat[(y_row == positive) | (y_row == negative)]
                    # 记录更改之前的坐标，防止 positive_label 和 negative 相等或 negative_label 和 positive 相等这种情况
                    positive_idx = y == positive
                    negative_idx = y == negative
                    if self.positive_label is not None:
                        y[positive_idx] = self.positive_label
                    if self.negative_label is not None:
                        y[negative_idx] = self.negative_label
                    learner.train(x, y)

    def predict(self, x_mat: ndarray) -> Union[ndarray, int]:
        """
        返回预测值，是对应于 x_mat 的标记。

        :param x_mat: 特征向量组，行数 m 表示样本数，列数 n 表示特征数
        :return: 预测标记
        """

        if not self._trained:
            raise StateError('not trained yet')

        x_mat = _t.r2m(x_mat)
        pred = self.__predict(x_mat)

        if self.strategy == 'ovr':
            return _t.ret(pred)
        else:
            m = x_mat.shape[0]
            result = np.empty((m,))
            for i in range(m):
                r = pred[:, i]
                result[i] = sorted([(np.sum(r == label), label) for label in set(r)])[-1][1]

            return _t.ret(result)

    # FIXME: 更改 cost，查看哪里出了问题
    def cost(self, x_mat: ndarray, y_row: ndarray) -> float:
        """
        计算在 x_mat 和 y_row 上的代价。此方法只计算参与了分类结果的学习器。

        :param x_mat: 特征向量组，行数 m 表示样本数，列数 n 表示特征数
        :param y_row: 输出行向量，每一个值代表 x_mat 中对应行的输出
        :return: 代价值。
        """

        if not self._trained:
            raise StateError('not trained yet')

        x_mat, y_row = _t.match_x_y(x_mat, y_row, add_ones=False)
        pred = self.__predict(x_mat)
        cost_sum = 0
        m = x_mat.shape[0]

        if self.strategy == 'ovr':
            for i, label in enumerate(pred):
                y = y_row[i]
                if y == label:
                    y = self.positive_label if self.positive_label is not None else label
                else:
                    y = self.negative_label if self.negative_label is not None else label + 1
                cost_sum = cost_sum + self._learners[label].cost(x_mat[i, :], np.array([y]))
        else:
            result = np.empty((m,))
            for i in range(m):
                r = pred[:, i]
                result[i] = sorted([(np.sum(r == label), label) for label in set(r)])[-1][1]

            for i in range(m):
                sub_cost_sum = 0
                count = 0
                for j, ((positive, negative), learner) in enumerate(self._learners.items()):
                    if pred[j][i] == result[i]:
                        y = y_row[i]
                        if y == positive:
                            y = self.positive_label if self.positive_label is not None else positive
                        else:
                            y = self.negative_label if self.negative_label is not None else negative
                        sub_cost_sum = sub_cost_sum + learner.cost(x_mat[i, :], np.array([y]))
                        count = count + 1
                if count != 0:
                    cost_sum = cost_sum + sub_cost_sum / count

        return cost_sum / m

    def confidence(self, x_mat: ndarray) -> ndarray:
        """
        返回对应于 x_mat 的预测置信度。返回一个m*num_labels 的矩阵，其中每一行表示样本在每个类上的置信度。
        需要注意的是，如果所使用的二分类器不支持 confidence 方法，那么会抛出异常。

        :param x_mat: 特征向量组，行数 m 表示样本数，列数 n 表示特征数
        :return: 预测概率。
        """

        return self.__evaluation(x_mat, lambda learner: learner.confidence(x_mat))

    def probability(self, x_mat: ndarray):
        """
        返回对应于 x_mat 的预测概率。返回一个m*num_labels 的矩阵，其中每一行表示样本在每个类上的概率。
        需要注意的是，如果所使用的二分类器不支持 probability 方法，那么会抛出异常。

        :param x_mat: 特征向量组，行数 m 表示样本数，列数 n 表示特征数
        :return: 预测概率。
        """

        return self.__evaluation(x_mat, lambda learner: learner.probability(x_mat))

    def __predict(self, x_mat: ndarray) -> ndarray:
        if self._strategy == 'ovr':
            confidences = np.empty((len(self._labels), x_mat.shape[0]))
            for i, learner in enumerate(self._learners.values()):
                confidences[i, :] = learner.confidence(x_mat)

            return self.labels[np.argmax(confidences, axis=0)]
        else:
            m = x_mat.shape[0]
            preds = np.empty((len(self._learners), m))

            for i, ((positive, negative), learner) in enumerate(self._learners.items()):
                pred = learner.predict(x_mat)
                # 结果数据类型可能是 bool，如果还对它赋值结果还会是 bool
                if pred.dtype == 'bool':
                    pred = pred.astype(dtype='i2')
                positive_label = positive if self.positive_label is None else self.positive_label
                negative_label = negative if self.negative_label is None else self.negative_label
                positive_idx = pred == positive_label
                negative_idx = pred == negative_label
                if self.positive_label is not None:
                    pred[positive_idx] = positive
                if self.negative_label is not None:
                    pred[negative_idx] = negative
                preds[i, :] = pred

            return preds

    def __evaluation(self, x_mat: ndarray, eval: Callable[[IProbabilityLearner], ndarray]):
        if not self._trained:
            raise StateError('not trained yet')

        x_mat = _t.r2m(x_mat)
        m = x_mat.shape[0]
        n = len(self._labels)
        evaluation = np.zeros((m, n), dtype=float)

        if self.strategy == 'ovr':
            for i, learner in enumerate(self._learners.values()):
                evaluation[:, i] = eval(learner)
        else:
            for i, positive in enumerate(self.labels):
                for negative in self.labels:
                    if positive == negative:
                        continue
                    evaluation[:, i] = evaluation[:, i] + eval(self._learners[(positive, negative)])
                evaluation[:, i] = evaluation[:, i] / (n - 1)

        return evaluation


def linear_error(predict_y: ndarray, actual_y: ndarray, probability: Optional[Callable[[int or float], float]] = None) \
        -> float:
    """
    求回归问题的均方误差。

    :param predict_y: 预测的结果
    :param actual_y: 实际的结果
    :param probability: 数据分布的概率密度函数。为 None 表示数据分布是均匀的
    :return: 均方误差
    """

    predict_y, actual_y = __check_predict_actual_y(predict_y, actual_y)
    m = predict_y.shape[0]
    if probability is None:
        return np.sum((predict_y - actual_y) ** 2) / m
    else:
        diff_sum = 0
        for x, a in zip(predict_y, actual_y):
            diff_sum = diff_sum + (x - a) ** 2 * probability(x)

        return diff_sum / m


def accuracy(predict_y: ndarray, actual_y: ndarray) -> float:
    """
    求分类问题的精确度。

    :param predict_y: 预测的结果
    :param actual_y: 实际的结果
    :return: 精确度
    """

    predict_y, actual_y = __check_predict_actual_y(predict_y, actual_y)
    # noinspection PyTypeChecker
    return np.mean(predict_y == actual_y)


def precision_and_recall(predict_y: ndarray, actual_y: ndarray, *,
                         labels: Union[int, Tuple[int], List[int], ndarray] = 1,
                         beta: Union[int, float] = 1) -> \
        Tuple[Dict[int, float], Dict[int, float], Dict[int, float], Tuple[float, float, float],
              Tuple[float, float, float]]:
    """
    计算查准率和查全率，以及它们的度量值 F。还有宏观和微观的值。

    :param predict_y: 预测的结果
    :param actual_y: 实际的结果
    :param labels: 需要计算查准率和查全率的标记。如果为 int 表示对某个标签进行计算，为 Iterable 表示对一组标签进行计算。默认为
                    1，表示计算二分类问题中值为 1 的标记
    :param beta: 度量值的权重，当它等于 1 时，就是 F1 值；大于 1 表示更重视查全率；小于 1 表示更重视查准率。默认为 1
    :return: 前三个返回值分别是查准率、查全率和 F 值字典，键是标签，值是查准率/查全率；后两个参数是两个元组，分别包含宏查准率、
             宏查全率、宏 F 值和微查准率、微查全率、微 F 值
    """

    predict_y, actual_y = __check_predict_actual_y(predict_y, actual_y)
    precisions = {}
    recalls = {}
    f = {}

    if isinstance(labels, int):
        labels = [labels]

    tp_sum = 0
    fp_sum = 0
    fn_sum = 0
    precision_sum = 0
    recall_sum = 0
    n = len(labels)
    for label in labels:
        tp = np.sum((predict_y == label) & (actual_y == label))
        fp = np.sum((predict_y == label) & (actual_y != label))
        fn = np.sum((predict_y != label) & (actual_y == label))

        precisions[label] = tp / (tp + fp)
        recalls[label] = tp / (tp + fn)
        f[label] = (1 + beta ** 2) * precisions[label] * recalls[label] / (beta ** 2 * precisions[label] +
                                                                           recalls[label])

        tp_sum = tp_sum + tp
        fp_sum = fp_sum + fp
        fn_sum = fn_sum + fn
        precision_sum = precision_sum + precisions[label]
        recall_sum = recall_sum + recalls[label]

    micro_tp = tp_sum / n
    micro_fp = fp_sum / n
    micro_fn = fn_sum / n
    micro_precision = micro_tp / (micro_tp + micro_fp)
    micro_recall = micro_tp / (micro_tp + micro_fn)
    micro_f = (1 + beta ** 2) * micro_precision * micro_recall / (beta ** 2 * micro_precision + micro_recall)

    macro_precision = precision_sum / n
    macro_recall = recall_sum / n
    macro_f = (1 + beta ** 2) * macro_precision * macro_recall / (beta ** 2 * macro_precision + macro_recall)

    return precisions, recalls, f, (macro_precision, macro_recall, macro_f), (micro_precision, micro_recall, micro_f)


def pr(positive_accuracy: ndarray, actual_y: ndarray, positive_label: int = 1) -> Tuple[ndarray, ndarray]:
    """
    计算 P-R 曲线值。

    :param positive_accuracy: 正例率，其中每一个正例率表示对应的样本值是正例的预测概率。
    :param actual_y: 实际的结果
    :param positive_label: 正例标记，表示将这个标记作为正例。默认为 1，表示计算二分类问题中值为 1 的标记
    :return: 第一个值是查准率；第二个值是对应的查全率
    """

    positive_accuracy, actual_y = __check_predict_actual_y(positive_accuracy, actual_y, name='positive_accuracy')
    indices = np.argsort(positive_accuracy)[::-1]
    positive_accuracy = positive_accuracy[indices]
    actual_y = actual_y[indices]

    m = positive_accuracy.shape[0]
    precisions = np.empty((m,))
    recalls = np.empty((m,))
    predict_y = np.empty((m,))
    for i in range(m):
        predict_y[0:i + 1] = positive_label
        predict_y[i + 1:] = positive_label + 1
        precision, recall, *_ = precision_and_recall(predict_y, actual_y, labels=positive_label)
        precisions[i] = precision[positive_label]
        recalls[i] = recall[positive_label]

    return precisions, recalls


def plot_pr(precisions: Union[ndarray, Tuple[float], List[float]], recalls: Union[ndarray, tuple, list], *,
            title: str = 'P-R Curve', xlabel: str = 'Recall Rate', ylabel: str = 'Precision Rate'):
    """
    绘制 P-R 曲线。

    :param precisions: 查准率
    :param recalls: 查全率
    :param title: 标题
    :param xlabel: x 轴标签
    :param ylabel: y 轴标签
    """

    plt.plot(recalls, precisions)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.axis([0, 1.03, 0, 1.03])
    plt.show()


def roc(positive_accuracy: ndarray, actual_y: ndarray, positive_label: int = 1) -> Tuple[ndarray, ndarray, float]:
    """
    计算 ROC 曲线值。

    :param positive_accuracy: 正例率，其中每一个正例率表示对应的样本值是正例的预测概率。
    :param actual_y: 实际的结果
    :param positive_label: 正例标记，表示将这个标记作为正例。默认为 1，表示计算二分类问题中值为 1 的标记
    :return: 第一个值是真正例率；第二个值是对应的假正例率；第三个值是 AUC
    """

    positive_accuracy, actual_y = __check_predict_actual_y(positive_accuracy, actual_y, name='positive_accuracy')
    indices = np.argsort(positive_accuracy)[::-1]
    positive_accuracy = positive_accuracy[indices]
    actual_y = actual_y[indices]

    m = positive_accuracy.shape[0]
    predict_y = np.empty((m,))
    tprs = np.zeros((m + 1,))  # 真正例率
    fprs = np.zeros((m + 1,))  # 假正例率
    for i in range(m):
        predict_y[positive_accuracy >= positive_accuracy[i]] = positive_label
        predict_y[positive_accuracy < positive_accuracy[i]] = positive_label + 1
        tp = np.sum((predict_y == positive_label) & (actual_y == positive_label))
        fp = np.sum((predict_y == positive_label) & (actual_y != positive_label))
        fn = np.sum((predict_y != positive_label) & (actual_y == positive_label))
        tn = np.sum((predict_y != positive_label) & (actual_y != positive_label))

        tprs[i + 1] = tp / (tp + fn)
        fprs[i + 1] = fp / (tn + fp)

    auc = 0
    for i in range(m):
        auc = auc + (fprs[i + 1] - fprs[i]) * (tprs[i] + tprs[i + 1])
    auc = auc / 2

    return tprs, fprs, auc


def plot_roc(tprs: Union[ndarray, Tuple[float], List[float]], fprs: Union[ndarray, tuple, list], *,
             title: str = 'ROC Curve', xlabel: str = 'False Positive Rate', ylabel: str = 'True Positive Rate'):
    """
    绘制 ROC 曲线。

    :param tprs: 真正例率
    :param fprs: 假正例率
    :param title: 标题
    :param xlabel: x 轴标签
    :param ylabel: y 轴标签
    """

    plt.plot(fprs, tprs)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.axis([-0.008, 1, 0, 1.03])
    plt.show()


def cost_sensitive(predict_y: ndarray, actual_y: ndarray, cost_sensitive_mat: Dict[int, Dict[int, Union[int, float]]]) \
        -> float:
    """
    计算代价敏感错误率。

    :param predict_y: 正例率，其中每一个正例率表示对应的样本值是正例的预测概率。
    :param actual_y: 实际的结果
    :param cost_sensitive_mat: 代价敏感矩阵。形式上是一个矩阵，行是实际标签，列是预测标签，值是将实际标签判断为预测标签的代价
    :return: 代价敏感错误率
    """

    predict_y, actual_y = __check_predict_actual_y(predict_y, actual_y)
    cost_sensitive_rate = 0
    for p, a in zip(predict_y, actual_y):
        cost_sensitive_rate = cost_sensitive_rate + cost_sensitive_mat[a][p]
    cost_sensitive_rate = cost_sensitive_rate / predict_y.shape[0]

    return cost_sensitive_rate


# TODO: 计算学习器的期望总体代价


# noinspection PyUnresolvedReferences
def cost_curve(x_mat: ndarray, y_vec: ndarray, x_cv: ndarray, y_cv: ndarray,
               learner: Callable[[], ISuperviseLearner],
               *, ran: Union[int, Tuple[int], List[int], ndarray] = 0, add_ones: bool = True) \
        -> Tuple[ndarray, ndarray]:
    """
    获取学习曲线值，诊断算法是否有高偏差或高方差问题。

    :param x_mat: 特征向量组，行数 m 表示样本数，列数 n 表示特征数
    :param y_vec: 输出向量，可以是列向量也可以是行向量，每一个值代表 x_mat 中对应行的输出
    :param x_cv: 交叉验证集特征向量组
    :param y_cv: 交叉验证集输出向量
    :param learner: 学习器生成函数，用以生成一个学习器，这个学习器必须有 train 方法用以训练，还要有 cost 方法，
                             用以计算代价
    :param ran: 需要检验的训练集样本范围，默认为 0，表示选取所有的训练集样本。
    :param add_ones: 是否增加截距项
    :return: 训练集代价行向量；验证集代价行向量
    """

    x_mat, y_vec = _t.match_x_y(x_mat, y_vec, add_ones=add_ones)
    x_cv, y_cv = _t.match_x_y(x_cv, y_cv, add_ones=add_ones)

    m = x_mat.shape[0]
    if isinstance(ran, int):
        if ran <= 0 or ran > m:
            ran = np.arange(1, m + 1)
        else:
            ran = np.arange(1, ran + 1)

    cost_train = np.empty((len(ran),))
    cost_cv = np.empty((len(ran),))
    for i, interval in enumerate(ran):
        if interval <= 0 or interval > m:
            continue
        xi = x_mat[:interval, :]
        yi = y_vec[:interval]
        lm = learner()
        lm.train(xi, yi)

        cost_train[i] = lm.cost(xi, yi)
        cost_cv[i] = lm.cost(x_cv, y_cv)

    return cost_train, cost_cv


# FIXME: 有 bug
# noinspection PyUnresolvedReferences
def regularization_curve(x_mat: ndarray, y_vec: ndarray, x_cv: ndarray, y_cv: ndarray,
                         learner: Callable[[Union[int, float]], ISuperviseLearner], *,
                         lambs: Union[tuple, list, ndarray] = (0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10)):
    """
    不断选取不同的正则化参数，验证在不同正则化参数下的误差。

    :param x_mat: 特征向量组，行数 m 表示样本数，列数 n 表示特征数
    :param y_vec: 输出向量，可以是列向量也可以是行向量，每一个值代表 x_mat 中对应行的输出
    :param x_cv: 交叉验证集特征向量组
    :param y_cv: 交叉验证集输出向量
    :param learner: 学习器生成函数，参数是正则化参数，用以生成一个学习器，这个学习器必须有 train 方法用以训练，
                             还要有 cost 方法，用以计算代价
    :param lambs: lamb 参数集
    :return: 训练集代价行向量；验证集代价行向量；lambs 参数
    """

    x_mat, y_vec = _t.match_x_y(x_mat, y_vec)
    x_cv, y_cv = _t.match_x_y(x_cv, y_cv)

    lambs_len = len(lambs)
    cost_train = np.empty((lambs_len,))
    cost_cv = np.empty((lambs_len,))

    for i, lamb in enumerate(lambs):
        lm = learner(lamb)
        lm.train(x_mat, y_vec)
        cost_train[i] = lm.cost(x_mat, y_vec)
        cost_cv[i] = lm.cost(x_cv, y_cv)

    return cost_train, cost_cv, lambs


def plot_cost_curve(cost_train: ndarray, cost_cv: ndarray, *, title: str = 'Cost Curve',
                    legend: Union[Tuple[str], List[str]] = ('Train', 'Cross Validation'),
                    ran: Union[int, Tuple[int], List[int], ndarray] = 0,
                    xlabel: str = 'Number of training examples', ylabel: str = 'Cost'):
    """
    绘制代价曲线。

    :param cost_train: 训练集误差
    :param cost_cv: 交叉验证集误差
    :param title: 标题，默认为 'Learning curve'
    :param legend: 训练集误差曲线和交叉验证集代价曲线的标题
    :param ran: 选取的样本数量范围，等于 0 的话就为 err_train 或 err_cv 的最大值
    :param xlabel: x 轴标签
    :param ylabel: y 轴标签
    """

    if isinstance(ran, int):
        if ran != 0:
            ran = np.arange(ran)
        else:
            ran = np.arange(max((cost_train.shape[0], cost_cv.shape[0])))
    max_x = ran[-1]
    tmax = np.max(cost_train)
    cmax = np.max(cost_cv)
    tmin = np.min(cost_train)
    cmin = np.min(cost_cv)
    min_y = tmin if tmin < cmin else cmin
    max_y = tmax if tmax > cmax else cmax
    pad = 1

    i1, = plt.plot(ran, cost_train)
    i2, = plt.plot(ran, cost_cv)
    plt.title(title)
    plt.legend([i1, i2], legend, loc='upper right')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.axis([0, max_x, min_y - pad, max_y + pad])
    plt.show()


def __check_predict_actual_y(predict_y: ndarray, actual_y: ndarray, *, name: str = 'predict_y') \
        -> Tuple[ndarray, ndarray]:
    predict_y = _t.c2r(predict_y)
    actual_y = _t.c2r(actual_y)
    if predict_y.shape[0] != actual_y.shape[0]:
        raise DataNotMatchError('number of %s and actual_y do not match' % name)

    return predict_y, actual_y
