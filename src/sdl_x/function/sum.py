import numpy as np

from .function import Function
from ..util_reshape_sum_backward import reshape_sum_backward
from .sum_and_brodcast_to import broadcast_to


class Sum(Function):
    def __init__(self, axis, keepdims):
        self.x_shape = None
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, x: np.ndarray):
        self.x_shape = x.shape
        y = x.sum(axis=self.axis, keepdims=self.keepdims)
        return y

    def backward(self, gy):
        gy = reshape_sum_backward(gy, self.x_shape, self.axis, self.keepdims)
        gx = broadcast_to(gy, self.x_shape)
        return gx


def sum_(x, axis=None, keepdims=False):
    return Sum(axis, keepdims)(x)
