import numpy as np

from sdl_x.exp import Exp
from sdl_x.square import Square
from sdl_x.variable import Variable


# noinspection DuplicatedCode
def test_backward():
    A = Square()
    B = Exp()
    C = Square()

    x = Variable(np.array(0.5))
    a = A(x)
    b = B(a)
    y = C(b)

    y.grad = np.array(1.0)
    y.backward()
    # 手动计算大概是 3.2974425
    assert np.allclose(x.grad.data, 3.2974425)


def test_operation():
    a = Variable(np.array(1))
    # Variable + np.array and vise
    b = a + np.array(20)
    assert b.data == 21
    b = np.array(20) + a
    assert b.data == 21

    # Variable + number and vise
    c = a * 20
    assert c.data == 20
    c = 20 * a
    assert c.data == 20


