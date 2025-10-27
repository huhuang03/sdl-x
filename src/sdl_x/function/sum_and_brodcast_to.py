import numpy as np

from .function import Function
from ..variable import as_variable


class BroadcastTo(Function):
    def __init__(self, shape):
        self.shape = shape
        self.x_shape = None

    def forward(self, x: np.ndarray):
        self.x_shape = x.shape
        y = np.broadcast_to(x, self.shape)
        return y

    def backward(self, gy: np.ndarray):
        gx = sum_to(gy, self.x_shape)
        return gx


def broadcast_to(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return BroadcastTo(shape)(x)


def _real_sum_to(arr: np.ndarray, shape):
    raise NotImplementedError()


class SumTo(Function):
    def __init__(self, shape):
        self.shape = shape
        self.x_shape = None

    def forward(self, x: np.ndarray):
        self.x_shape = x.shape
        return _real_sum_to(x, self.shape)

    def backward(self, gy: np.ndarray):
        gx = broadcast_to(gy, self.x_shape)
        return gx


def sum_to(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return SumTo(shape)(x)
