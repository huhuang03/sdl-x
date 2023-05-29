import numpy as np
from abc import ABC, abstractmethod


class ICommonLayer(ABC):
    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def backward(self, dout: np.ndarray) -> np.ndarray:
        pass

class ILossLayer(ABC):
    def __init__(self):
        self.t = None
        self.y = None

    @abstractmethod
    def forward(self, y: np.ndarray, t: np.ndarray) -> float:
        pass

    @abstractmethod
    def backward(self) -> np.ndarray:
        pass