import numpy as np

from src.sdl_x.numerical_diff import numerical_diff
from src.sdl_x.square import Square
from src.sdl_x.variable import Variable


def test_numerical_diff():
    f = Square()
    x = Variable(np.array(2.0))
    dy = numerical_diff(f, x)
    assert (dy - 4.0) < 1e-5