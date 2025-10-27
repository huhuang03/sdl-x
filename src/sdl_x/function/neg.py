import numpy as np

from .function import Function
from ..variable import Variable


class Neg(Function):
    def forward(self, x: np.ndarray):
        return -x

    def backward(self, gy: np.ndarray):
        return gy * -1


def neg(x: Variable):
    return Neg()(x)