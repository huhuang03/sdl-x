import numpy as np

from .function import Function


class Transpose(Function):
    def forward(self, x):
        return np.transpose(x)
        pass

    def backward(self, gy):
        return transpose(gy)


def transpose(x):
    return Transpose()(x)