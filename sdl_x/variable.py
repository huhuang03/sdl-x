from typing import Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from sdl_x.function.function import Function


def _check_data(data):
    if data is not None:
        if not isinstance(data, np.ndarray):
            raise TypeError('{} is not supported'.format(type(data)))


class Variable:
    __array_priority__ = 20

    def __init__(self, data: np.ndarray, name=None):
        _check_data(data)
        # is this always np.ndarray??
        self.data: np.ndarray = data
        self.grad: Optional[Variable] = None
        self.name = name
        self.creator: Optional['Function'] = None
        self.generation = 0

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

    def __repr__(self):
        if self.data is None:
            return 'variable(None)'
        p = str(self.data).replace('\n', '\n' + ' ' * 9)
        return f'variable({p})'

    def __len__(self):
        return len(self.data)

    def __neg__(self):
        raise NotImplementedError()

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
            self.grad = Variable(np.ones_like(self.data))
        funcs = []

        def add_func(_func: 'Function'):
            funcs.append(_func)
            funcs.sort(key=lambda it: it.generation)

        add_func(self.creator)

        while funcs:
            func = funcs.pop()
            inputs = func.inputs
            outputs = func.outputs
            input_grads = func.backward(*[i.grad for i in outputs])
            if not isinstance(input_grads, tuple):
                input_grads = (input_grads,)
            for input_, grad in zip(inputs, input_grads):
                # 这里我觉得，应该只能是 + 或者x
                if input_.grad is not None:
                    input_.grad = grad + input_.grad
                else:
                    input_.grad = grad
                if input_.creator:
                    add_func(input_.creator)


def as_variable(val):
    if isinstance(val, Variable):
        return val
    return Variable(val)
