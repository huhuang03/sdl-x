from .function import Function
from sdl_x.util import as_array


class Mul(Function):

    def forward(self, x0, x1):
        return x0 * x1

    def backward(self, gy):
        x0, x1 = self.inputs
        return x1 * gy, x0 * gy


def mul(x0, x1):
    return Mul()(x0, as_array(x1))