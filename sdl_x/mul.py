from .function import Function
from .util import as_array


class Mul(Function):

    def forward(self, x0, x1):
        return x0 * x1

    def backward(self, gy):
        return self.inputs[1].data * gy, self.inputs[0].data * gy


def mul(x0, x1):
    return Mul()(x0, as_array(x1))