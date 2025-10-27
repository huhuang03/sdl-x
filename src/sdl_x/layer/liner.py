import numpy as np
from .layer import Layer
from ..parameter import Parameter


class Liner(Layer):
    def __init__(self, out_size, nobias=False, dtype=np.float32, in_size=None):
        super().__init__()
        self.dtype = dtype
        self.in_size = in_size
        self.out_size = out_size
        I, O = in_size, out_size
        if in_size is not None:
            self._init_W()
        if nobias:
            self.b = None
        else:
            self.b = Parameter(np.zeros(O, dtype=dtype), name='b')

    def _init_W(self):
        I, O = self.in_size, self.out_size
        W_data = np.random.rand(I, O).astype(self.dtype) * np.sqrt(1 / I)
        self.W = Parameter(W_data, name='W')


    def forward(self, x):
        if self.W.data is None:
            self.in_size = x.shape[1]
            self._init_W()
        y = linear(x, self.W, self.b)
        return y
