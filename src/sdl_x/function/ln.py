import numpy as np

from src.sdl_x.function.function import Function


class Ln(Function):
    def forward(self, x: np.ndarray):
        return np.log(x)

    def backward(self, gy):
        # 这里有问题？
        return 1/self.inputs[0] * gy


def ln(x):
    return Ln()(x)
