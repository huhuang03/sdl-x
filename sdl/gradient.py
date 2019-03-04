# encoding=utf-8
import numpy as np


def _numerical_gradient_1d(f, x):
    h = 1e-4
    grad = np.zeros_like(x)
    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = tmp_val + h
        fxh1 = f(x)

        x[idx] = tmp_val - h
        fxh2 = f(x)
        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp_val
    return grad


def numerical_gradient(f, x):
    if x.ndim == 1:
        return _numerical_gradient_1d(f, x)
    else:
        # 这里使用了单行x去计算f。实际应该是整个x参与计算。
        grad = np.zeros_like(x)
        for i in range(x.shape[0]):
            grad[i] = numerical_gradient(f, x[i])
        return grad


def gradient_decline(f, init_x, learn_gap ,step = 10000):
    """梯度下降，可能单词用错了"""
    x = init_x
    for i in range(step):
        grad = numerical_gradient(f, x)
        x = x - grad * learn_gap
    return f(x)
