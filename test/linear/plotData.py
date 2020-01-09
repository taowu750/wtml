#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from matplotlib import pyplot as plt


def plot_data(x, y):
    plt.xlabel('Population of City in 10,000s')
    plt.ylabel('Profit in $10,000s')
    plt.plot(x, y, 'rx', markersize=6)
    plt.show()
