from typing import List, Union
from abc import ABC, abstractmethod
from .variable import Variable, as_variable
from .config import Config
from .util import as_array


class Function(ABC):
    def __call__(self, *inputs: Variable) -> Union[List[Variable], Variable]:
        self.inputs = [as_variable(it) for it in inputs]
        y = self.forward(*[x.data for x in inputs])
        if not isinstance(y, tuple):
            y = (y,)
        self.outputs = [Variable(as_array(i)) for i in y]
        if Config.enable_backprop:
            self.generation = max([it.generation for it in self.inputs])
            for i in self.outputs:
                i.set_creator(self)
        return self.outputs if len(self.outputs) > 1 else self.outputs[0]

    @abstractmethod
    def forward(self, *x):
        """
        Returns:
            tuple or single object
        """
        raise NotImplemented()

    @abstractmethod
    def backward(self, *gy):
        """
        in: gy
        out: f'(x) * gy
        Returns:
            tuple of single object
        """
        raise NotImplemented
