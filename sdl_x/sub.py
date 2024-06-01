from .function import Function
from .util import as_array


class Sub(Function):
    def forward(self, x0, x1):
        return x0 - x1,

    def backward(self, gy: int):
        return gy, -gy


def sub(x0, x1):
    return Sub()(x0, as_array(x1))


def rsub(x0, x1):
    return Sub()(as_array(x1), x0)
