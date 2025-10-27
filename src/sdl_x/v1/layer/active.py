# active layers
import numpy as np
from src.sdl_x import softmax

from .old_layer import ICommonLayer


class SoftMax(ICommonLayer):

    def forward(self, x: np.ndarray) -> np.ndarray:
        return softmax(x)

    def backward(self, dout: np.ndarray) -> np.ndarray:
        pass