from .function import Function
from sdl_x.util import as_array
import numpy as np


class Pow(Function):
    def forward(self, x0: int, x1: int):
        return x0 ** x1

    def backward(self, gy):
        x0, x1 = [i.data for i in self.inputs]
        return (x1 * (x0 ** (x1 - 1))) * gy, (x0 ** x1) * np.log(x0) * gy


def pow_(x0, x1):
    return Pow()(x0, as_array(x1))
