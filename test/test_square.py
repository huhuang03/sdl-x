import numpy as np

from src.sdl_x import Square
from src.sdl_x import Variable


def test_square():
    s = Square()
    rst = s(Variable(np.array([1])))
    assert rst.data == 1

    rst = s(Variable(np.array([2])))
    assert rst.data == 4
