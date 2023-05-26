# active layers
import numpy as np
from sdl_x import softmax

from .layer import ICommonLayer


class SoftMax(ICommonLayer):

    def forward(self, x: np.ndarray) -> np.ndarray:
        return softmax(x)

    def backward(self, dout: np.ndarray) -> np.ndarray:
        pass

