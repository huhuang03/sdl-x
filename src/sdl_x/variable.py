from typing import Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .function.function import Function


def _check_data(data):
    if data is not None:
        if not isinstance(data, np.ndarray):
            raise TypeError('{} is not supported'.format(type(data)))


class Variable:
    __array_priority__ = 20

    def __init__(self, data: np.ndarray, name=None):
        _check_data(data)
        self.data: np.ndarray = data
        self.grad: Optional[np.ndarray] = None
        self.name = name
        # the function that create this variable
        self.creator: Optional['Function'] = None
        self.generation = 0

    def real_sum(self, axios, keepdims):
        raise NotImplementedError('Variable.sum() not implemented')

    def sum(self, axios=None, keepdims=False):
        self.sum(axios, keepdims)

    def __pow__(self, power, modulo=None):
        raise NotImplementedError()

    def __rmul__(self, other) -> 'Variable':
        raise NotImplementedError()

    def __mul__(self, other) -> 'Variable':
        raise NotImplementedError()

    def __add__(self, other) -> 'Variable':
        raise NotImplementedError

    def __radd__(self, other) -> 'Variable':
        raise NotImplementedError()

    def __sub__(self, other) -> 'Variable':
        raise NotImplementedError

    def __rsub__(self, other) -> 'Variable':
        raise NotImplementedError()

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def size(self):
        return self.data.size

    def transpose(self):
        raise NotImplementedError()

    @property
    def T(self):
        return self.transpose()

    def __repr__(self):
        if self.data is None:
            return 'variable(None)'
        p = str(self.data).replace('\n', '\n' + ' ' * 9)
        return f'variable({p})'

    def __len__(self):
        return len(self.data)

    def __neg__(self):
        raise NotImplementedError()

    def real_reshape(self, shape):
        raise NotImplementedError()

    def reshape(self, *shape):
        """
        how to hande recycle import?
        """
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        return self.real_reshape(shape)

    @property
    def dtype(self):
        return self.data.dtype

    def clear_grad(self):
        self.grad = None

    def set_creator(self, func: 'Function'):
        self.creator = func
        self.generation = func.generation

    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)
        """
        is created back graphic
        """
        f = self.creator
        if f is not None:
            x_variable = f.inputs
            x_variable.grad = f.backward(self.grad)
            x_variable.backward()


def as_variable(val: Variable | np.ndarray):
    if isinstance(val, Variable):
        return val
    return Variable(val)
