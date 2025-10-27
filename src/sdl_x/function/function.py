from typing import List, Union
from abc import ABC, abstractmethod

import numpy as np

from sdl_x.variable import Variable, as_variable
from sdl_x.config import Config
from sdl_x.util import as_array


class Function(ABC):
    def __call__(self, inputs: Variable) -> Variable:
        self.inputs = inputs
        x = inputs.data
        y = self.forward(x)
        output = Variable(y)
        output.creator = self
        self.output = output
        return output
        # self.inputs = inputs
        # y = self.forward(inputs.data)
        # y.creator = self
        # return Variable(y)
        # if not isinstance(y, tuple):
        #     y = (y,)
        # if Config.enable_backprop:
        #     self.generation = max([it.generation for it in self.inputs])
        #     for i in self.outputs:
        #         i.set_creator(self)
        # return self.outputs if len(self.outputs) > 1 else self.outputs[0]

    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Returns:
            tuple or single object
        """
        raise NotImplemented()

    @abstractmethod
    def backward(self, gy: np.ndarray) -> np.ndarray:
        """
        in: gy
        out: f'(x) * gy
        Returns:
            tuple of single object
        """
        raise NotImplemented
