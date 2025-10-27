from .function.function import Function
import numpy as np


class Exp(Function):
    """
    y = e^x
    """
    def forward(self, x):
        return np.exp(x)

    def backward(self, gy: np.ndarray):
        return np.exp(self.inputs.data) * gy


def exp(x):
    return Exp()(x)
