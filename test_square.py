from sdl_x.square import Square
from sdl_x.variable import Variable


def test_square():
    s = Square()
    rst = s(Variable(1))
    assert rst.data == 1

    rst = s(Variable(2))
    assert rst.data == 4
