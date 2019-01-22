# encoding=utf-8
import numpy as np

# x为一维数组
def _numerical_gradient1(f, x):
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


# x为二维维数组
def _numerical_gradient2(f, x):
    h = 1e-4
    grad = np.zeros_like(x)

    height, width = grad.shape

    for i in range(height):
        for j in range(width):
            tmp_val = x[i, j]
            x[i, j] = tmp_val + h
            fxh1 = f(x)

            x[i, j] = tmp_val -h
            fxh2 = f(x)
            grad[i, j] = (fxh1 - fxh2) / (2 * h)
            x[i, j] = tmp_val
    return grad


def book_gradient(f, x):
    """书上的实现"""
    h = 1e-4  # 0.0001
    grad = np.zeros_like(x)

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)  # f(x+h)

        x[idx] = tmp_val - h
        fxh2 = f(x)  # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2 * h)

        x[idx] = tmp_val  # 还原值
        it.iternext()

    return grad


def numerical_gradient(f, x):
    if x.ndim == 1:
        return _numerical_gradient1(f, x)
    elif x.ndim == 2:
        return _numerical_gradient2(f, x)
    else:
        raise ValueError("wrong dimen")
