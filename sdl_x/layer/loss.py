import numpy as np
from sdl_x import cross_entropy_error

from .layer import ILossLayer


# noinspection PyPep8Naming
class CrossEntropyError(ILossLayer):
    def forward(self, y: np.ndarray, t: np.ndarray) -> np.ndarray:
        """
        :param y: y is all no zero. if is zero, can't calc the gradient
        :param t: t is not-oneshot. so value is something like [0, 0, ..., 1, 0, ...]
        """
        self.t = t
        self.y = y
        if self.t.shape != self.y.shape:
            N = self.y.shape[0]
            if t.shape[0] != N:
                raise ValueError(f"t is not compatible to y. t shape: {t.shape}, y shape: {y.shape}")
            if len(t.shape) != 1:
                raise ValueError(f"t is not compatible to y. t shape: {t.shape}, y shape: {y.shape}")
            new_t = np.zeros_like(y)
            for i in range(N):
                new_t[i][self.t[i]] = self.t[i]
            self.t = new_t

        print(f'y: {y}')
        print(f't: {self.t}')
        return np.array([cross_entropy_error(y, t)])

    def backward(self) -> np.ndarray:
        return -self.t * (1 / self.y)