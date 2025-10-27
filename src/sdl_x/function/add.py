import numpy as np

from .function import Function
from sdl_x.util import as_array
from .sum_and_brodcast_to import sum_to


class Add(Function):
    def __init__(self):
        super().__init__()
        self.x0_shape = None
        self.x1_shape = None

    def forward(self, x0: np.ndarray, x1: np.ndarray):
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        return x0 + x1

    def backward(self, gy):
        gx0, gx1 = gy, gy
        if self.x0_shape != self.x1_shape:
            gx0 = sum_to(gx0, self.x0_shape)
            gx1 = sum_to(gx1, self.x1_shape)
        return gx0, gx1


def add(x0, x1):
    return Add()(x0, as_array(x1))