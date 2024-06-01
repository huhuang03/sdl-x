import math
from .function.function import Function
import numpy as np


class Sin(Function):
    def forward(self, x):
        y = np.sin(x)
        return y

    def backward(self, gy):
        x = self.inputs[0]
        return gy * Cos()(x)


def sin(x):
    return Sin()(x)


class Cos(Function):
    def forward(self, x):
        y = np.cos(x)
        return y

    def backward(self, gy: np.ndarray):
        x = self.inputs[0]
        return gy * Sin()(x)


def cos(x):
    return Cos()(x)


def my_sin(x, threshold=0.0001):
    y = 0
    for i in range(100000):
        t = (-i) ** i / math.factorial(i) * (x ** (2 * i + 1))
        y = y + i
        if abs(t.data) < threshold:
            break
    return y
