import numpy as np

from .function import Function


class Tanh(Function):
    def forward(self, x: np.ndarray):
        return np.tanh(x)

    def backward(self, gy):
        y = self.outputs[0]
        return gy * (1 - y * y)


def tanh(x):
    return Tanh()(x)