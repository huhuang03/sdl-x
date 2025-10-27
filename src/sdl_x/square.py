import numpy as np

from .function.function import Function
from .variable import Variable


class Square(Function):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return x ** 2

    def backward(self, gy: np.ndarray) -> np.ndarray:
        print('self.inputs.data: ', self.inputs.data, type(self.inputs.data))
        return self.inputs.data * 2 * gy


def square(x: Variable):
    return Square()(x)
