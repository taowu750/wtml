#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
包含基础设施的模块
"""


from abc import ABCMeta, abstractmethod, ABC

from numpy import ndarray


# TODO: 在基类中添加训练状态的测试；添加类别属性
class ISuperviseLearner(metaclass=ABCMeta):
    """
    基础监督分类学习器，定义了监督学习器应该具有的方法
    """

    @abstractmethod
    def train(self, x_mat: ndarray, y_row: ndarray):
        """
        使用数据进行训练。

        :param x_mat: x_mat: 特征向量组，行数 m 表示样本数，列数 n 表示特征数
        :param y_row: 输出向量，可以是列向量也可以是行向量，每一个值代表 x_mat 中对应行的输出
        :return: 继承类可以返回一些训练后的结果，比如权值向量等
        """
        pass

    @abstractmethod
    def predict(self, x_mat: ndarray):
        """
        对数据进行预测，返回结果。

        :param x_mat: 特征向量组，行数 m 表示样本数，列数 n 表示特征数
        :return: 预测结果，可以是标记或实值
        """
        pass

    @abstractmethod
    def cost(self, x_mat: ndarray, y_row: ndarray) -> float:
        """
        计算在给定数据上的代价值。约定代价值越小越好。

        :param x_mat: 特征向量组，行数 m 表示样本数，列数 n 表示特征数
        :param y_row: 输出向量，可以是列向量也可以是行向量，每一个值代表 x_mat 中对应行的输出
        :return: 代价值
        """
        pass


class IConfidenceLearner(ABC, ISuperviseLearner):
    """
    能给出置信度的监督分类学习器。置信度可以是概率或其他反映结果可信程度的值，
    置信度越大，表示结果越可信。
    在二分类问题中，给出正例置信度；在多分类问题中，为每个类别给出一个置信度。
    """

    @abstractmethod
    def confidence(self, x_mat: ndarray):
        """
        对数据进行预测，返回预测置信度。在二分类问题中，给出正例置信度；在多分类问题中，为每个类别给出一个置信度。

        :param x_mat: 特征向量组，行数 m 表示样本数，列数 n 表示特征数
        :return: 预测置信度。
        """
        pass


class IProbabilityLearner(IConfidenceLearner):
    """
    能够给出概率值的监督分类学习器。在二分类问题中，给出正例概率；在多分类问题中，为每个类别
    给出一个概率。
    """

    @abstractmethod
    def probability(self, x_mat: ndarray):
        """
        对数据进行预测，返回预测概率。在二分类问题中，给出正例概率；在多分类问题中，为每个类别
        给出一个概率。

        :param x_mat: 特征向量组，行数 m 表示样本数，列数 n 表示特征数
        :return: 预测概率，取值范围 [0, 1]
        """
        pass

    def confidence(self, x_mat: ndarray):
        """
        默认实现，采用预测概率作为置信度。

        :param x_mat: 特征向量组，行数 m 表示样本数，列数 n 表示特征数
        :return: 预测概率，取值范围 [0, 1]
        """

        return self.probability(x_mat)
