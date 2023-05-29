import numpy as np
from sdl_x import cross_entropy_error

from .layer import ILossLayer

class CrossEntropyError(ILossLayer):
    def forward(self, y: np.ndarray, t: np.ndarray) -> np.ndarray:
        """
        :param y: y is all no zero. if is zero, can't calc the gradient
        :param t: t is not-oneshot. so value is something like [0, 0, ..., 1, 0, ...]
        """
        self.t = t
        self.y = y
        return np.array([cross_entropy_error(y, t)])

    def backward(self) -> np.ndarray:
        return -self.t * (1 / self.y)