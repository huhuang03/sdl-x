from .function import Function


class Square(Function):
    def forward(self, x: float):
        return x ** 2

    def backward(self, gy):
        return self.inputs[0] * 2 * gy


def square(x):
    return Square()(x)