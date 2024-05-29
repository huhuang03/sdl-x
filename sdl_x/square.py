from .function import Function


class Square(Function):
    def forward(self, x):
        return x ** 2

    def backward(self, gy):
        return self.input_.data * 2 * gy
