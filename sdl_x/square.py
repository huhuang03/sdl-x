import numpy as np

from .function.function import Function


class Square(Function):
    def forward(self, x: float):
        return x ** 2

    def backward(self, gy: np.ndarray):
        return self.inputs[0] * 2 * gy


def square(x):
    return Square()(x)