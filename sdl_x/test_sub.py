from sdl_x.variable import Variable
import numpy as np


def test_sub():
    x = Variable(np.array(1))
    y = x - 3
    assert y.data == -2