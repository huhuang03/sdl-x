import numpy as np

from .function import Function
from sdl_x.util import as_array
from ..variable import Variable


class Mul(Function):

    def forward(self, x: np.ndarray) -> np.ndarray:
        x0, x1 = x[0], x[1]
        return x0 * x1

    def backward(self, gy):
        x0, x1 = self.inputs
        return x1 * gy, x0 * gy


def mul(x0, x1):
    return Mul()(Variable(np.ndarray([x0, x1])))