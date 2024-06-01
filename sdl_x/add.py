from .function import Function
from .util import as_array


class Add(Function):
    def forward(self, x0, x1):
        return x0 + x1,

    def backward(self, gy):
        return gy, gy


def add(x0, x1):
    return Add()(x0, as_array(x1))