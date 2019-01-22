# encoding=utf-8
import numpy as np
# this define some calculate loss functin

# 均方误差
def mean_squared_error(y, t):
    return 0.5 * np.sum((y - t) * 2)

# 交叉熵误差
## 监督数据是one-hot的形式
def cross_entropy_error_one_shot(y, t):
    delta = 1e-7
    if y.ndim == 1:
        y = y.reshape(1, y.size)
        t = t.reshape(1, t.size)

    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + delta)) / batch_size

## 监督数据是标签形式
def cross_entropy_error_label(y, t):
    delta = 1e-7
    if y.ndim == 1:
        y = y.reshape(1, y.size)
        t = t.reshape(1, t.size)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size


def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    # 监督数据是one-hot-vector的情况下，转换为正确解标签的索引
    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size
    # print("ndim: " + str(t.ndim))
    # if t.ndim == 1:
    #     return cross_entropy_error_label(y, t)
    # else:
    #     return cross_entropy_error_one_shot(y, t)
