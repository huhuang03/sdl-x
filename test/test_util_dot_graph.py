from sdl_x.util_dot_graph import _dot_var
from sdl_x.variable import Variable
from sdl_x.util_dot_graph import plot_dot_graph
import numpy as np


def test__dot_var():
    x = Variable(np.random.randn(2, 3))
    x.name = 'x'
    print(_dot_var(x))
    print(_dot_var(x, verbose=True))


# noinspection DuplicatedCode
def test_plot_graph():
    x = Variable(np.array(1))
    y = Variable(np.array(1))
    z = ((1 + (x + y + 1)**2 * (19 - 14*x + 3*x**2 - 14*y + 6*x*y + 3*y**2)) *
         (30 + (2*x - 3*y)**2 * (18 - 32*x + 12*x**2 + 48*y - 36*x*y + 27*y**2)))
    x.name = 'x'
    y.name = 'y'
    plot_dot_graph(z, verbose=False, to_file='../src/sdl_x/tmp_goldstein.png')