import numpy as np

from src.sdl_x.variable import Variable
from src.sdl_x.util_dot_graph import plot_dot_graph


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
    y.name = 'y'
    plot_dot_graph(y, to_file='tmp_0.png')
    y.backward()
    assert np.allclose(x.grad.data, 24)

    y.grad.name = 'gy'
    x.grad.name = 'gx'
    plot_dot_graph(x.grad, to_file='tmp_1.png')

    gx = x.grad
    x.clear_grad()
    print('gx: ', gx)

    gx.backward()
    assert np.allclose(x.grad.data, 44)

    # plt
    x.name = 'x'
    x.grad.name = "f''(x)"
    plot_dot_graph(x.grad, to_file='tmp_2.png')
