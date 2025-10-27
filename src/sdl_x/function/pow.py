from .function import Function
from src.sdl_x.util import as_array
from src.sdl_x.function.ln import ln


class Pow(Function):
    def forward(self, x0: int, x1: int):
        return x0 ** x1

    def backward(self, gy):
        x0, x1 = self.inputs
        return (x1 * (x0 ** (x1 - 1))) * gy, (x0 ** x1) * ln(x0) * gy


def pow_(x0, x1):
    return Pow()(x0, as_array(x1))
