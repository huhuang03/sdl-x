import numpy as np

from sdl_x.exp import Exp
from sdl_x.square import Square
from sdl_x.variable import Variable


# noinspection DuplicatedCode
def test_backward():
    A = Square()
    B = Exp()
    C = Square()

    x = Variable(np.array([0.5]))
    a = A(x)
    b = B(a)
    y = C(b)

    y.grad = np.array([1.0])
    print(x.grad)

    assert abs(x.grad - 3.2974425) < 1e-4

