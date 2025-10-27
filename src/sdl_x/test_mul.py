import numpy as np
from src.sdl_x.variable import Variable


def test_mul():
    x = Variable(np.array(3.0))
    y = Variable(np.array(4.0))

    k = x * y
    print(k.data == 12)
