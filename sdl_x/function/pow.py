from .function import Function
from sdl_x.util import as_array


class Pow(Function):
    def forward(self, x0: int, x1: int):
        return x0 ** x1

    def backward(self, gy):
        x = self.inputs[0]
        return 2 * x.data * gy


def pow_(x0, x1):
    return Pow()(x0, as_array(x1))
