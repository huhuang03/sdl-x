from typing import Optional, TYPE_CHECKING
if TYPE_CHECKING:
    from sdl_x.function import Function


class Variable:
    def __init__(self, data):
        self.data = data
        self.grad = None
        self.creator: Optional['Function'] = None

    def set_creator(self, func: 'Function'):
        self.creator = func

    def backward(self):
        f = self.creator
        if f is not None:
            x = f.input_
            x.grad = f.backward(self.grad)
            x.backward()