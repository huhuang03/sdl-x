# 33.2 手动求导，但是二阶
from sdl_x import Variable
import numpy as np


def f(x_: Variable) -> Variable:
    return x_ ** 4 - 2 * x_ ** 2


x = Variable(np.array(2.0), name='x')

lr = 0.001
iters = 10
# iters = 1

for i in range(iters):
    print(i, x)
    y = f(x)
    x.clear_grad()
    y.backward()

    gx = x.grad
    x.clear_grad()
    gx.backward()
    gx2 = x.grad

    x.data -= gx.data / gx2.data
