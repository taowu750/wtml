#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
此机器学习库的异常模块。
"""


class DataNotMatchError(Exception):
    """
    学习算法中数据不匹配时抛出的异常，比如训练集和标记数目不匹配等。
    """

    pass


class StateError(Exception):
    """
    当处于不正确的状态时，抛出此异常，比如数据未初始化等。
    """

    pass


class UnSupportError(Exception):
    """
    当使用了不受支持的操作时，抛出此异常，比如调用了为实现的基类方法等。
    """
