from sdl_x.variable import Variable
import numpy as np


def test_by_sphere():
    def sphere(_x, _y):
        return _x ** 2 + _y ** 2

    x = Variable(np.array(1))
    y = Variable(np.array(1))
    z = sphere(x, y)
    z.backward()
    assert x.grad.data == np.array(2)
    assert y.grad.data == np.array(2)


def test_by_matyas():
    x = Variable(np.array(1))
    y = Variable(np.array(1))
    z = 0.26 * (x ** 2 + y ** 2) - 0.48 * x * y
    z.backward()
    assert np.allclose(x.grad.data, 0.04)
    assert np.allclose(y.grad.data, 0.04)


def test_by_goldstein():
    # noinspection DuplicatedCode
    x = Variable(np.array(1))
    y = Variable(np.array(1))
    z = ((1 + (x + y + 1)**2 * (19 - 14*x + 3*x**2 - 14*y + 6*x*y + 3*y**2)) *
         (30 + (2*x - 3*y)**2 * (18 - 32*x + 12*x**2 + 48*y - 36*x*y + 27*y**2)))
    z.backward()
    assert x.grad.data == -5376
    assert y.grad.data == 8064
