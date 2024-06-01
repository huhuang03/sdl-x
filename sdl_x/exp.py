from .function import Function
import numpy as np


class Exp(Function):
    def forward(self, x):
        return np.exp(x)

    def backward(self, gy):
        return np.exp(self.inputs[0].data) * gy


def exp(x):
    return Exp()(x)
