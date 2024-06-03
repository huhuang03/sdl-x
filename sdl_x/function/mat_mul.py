import numpy as np

from .function import Function


# noinspection PyPep8Naming
class MatMul(Function):
    def forward(self, x: np.ndarray, W: np.ndarray):
        return x.dot(W)

    def backward(self, gy):
        x, W = self.inputs
        gx = matmul(gy, W.T)
        gW = matmul(x.T, gy)
        return gx, gW


# noinspection PyPep8Naming
def matmul(x, W):
    return MatMul()(x, W)
