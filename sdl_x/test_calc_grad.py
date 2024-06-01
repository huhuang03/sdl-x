import numpy as np

from sdl_x.variable import Variable
from sdl_x.util_dot_graph import plot_dot_graph


def test_calc_grad():
    # noinspection DuplicatedCode
    def f(x_) -> Variable:
        return x_ ** 4 - 2 * x_ ** 2

    x = Variable(np.array(2.0), name='x')
    y = f(x)
    y.backward()
    assert np.allclose(x.grad.data, 24)


def test_grad_twice():
    # noinspection DuplicatedCode
    def f(x_) -> Variable:
        return x_ ** 4 - 2 * x_ ** 2

    x = Variable(np.array(2.0), name='x')
    y = f(x)
    y.backward()
    assert np.allclose(x.grad.data, 24)
    gx = x.grad
    x.clear_grad()
    plot_dot_graph(gx)
    gx.backward()
    print('x.grad: ', x.grad)
