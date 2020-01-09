#!/usr/bin/env python
# -*- coding: utf-8 -*-


import matplotlib.pyplot as plt
import numpy as np


def display_data(X):
    m, n = X.shape
    # 计算高度宽度
    example_width = int(np.round(np.sqrt(X.shape[1])))
    example_height = n // example_width
    # 计算显示的图片行数和列数
    display_rows = int(np.floor(np.sqrt(m)))
    display_cols = int(np.ceil(m / display_rows))

    # 图片之间的边距
    pad = 1

    # 设置空白显示块
    display_array = -np.ones((pad + display_rows * (example_height + pad),
                              pad + display_cols * (example_width + pad)))

    # 将每个示例复制到显示块上的对应位置中
    curr_ex = 0
    for i in range(0, display_rows):
        for j in range(0, display_cols):
            if curr_ex >= m:
                break
            max_val = np.max(np.abs(X[curr_ex, :]))
            row_start = pad + i * (example_height + pad)
            col_start = pad + j * (example_width + pad)
            # matlab reshape 是按列的，numpy 是按行的
            display_array[row_start:row_start + example_height, col_start:col_start + example_width] = \
                X[curr_ex, :].reshape((example_width, example_height)).T / max_val
            curr_ex += 1
        if curr_ex >= m:
            break

    # 设置灰度图片并显示
    plt.imshow(display_array, cmap='gray')
    # 不显示坐标轴，需要在 imshow 之后，show 之前调用才有效
    plt.axis('off')
    plt.show()
