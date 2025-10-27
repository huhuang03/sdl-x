import numpy as np

from .function import Function
from sdl_x.util import as_array
from sdl_x.function.ln import ln
from ..variable import Variable


class Pow(Function):
    # you are strange!!!
    def forward(self, x: np.ndarray) -> np.ndarray:
        x0 = x[0]
        x1 = x[1]
        return x0 ** x1

    def backward(self, gy):
        x0, x1 = self.inputs
        return (x1 * (x0 ** (x1 - 1))) * gy, (x0 ** x1) * ln(x0) * gy


def pow_(x0, x1):
    return Pow()(Variable(np.array([x0, x1])))
