# 28.2 手动求导
from sdl_x.variable import Variable
import numpy as np


def rosenbrock(x0_: Variable, x1_: Variable) -> Variable:
    return 100 * (x1_ - x0_ ** 2) ** 2 + (x0_ - 1) ** 2


x0 = Variable(np.array(0.0), name='x')
x1 = Variable(np.array(2.0), name='y')

y = rosenbrock(x0, x1)
y.backward()

lr = 0.001
iters = 10000
# iters = 1

for i in range(iters):
    y = rosenbrock(x0, x1)
    x0.clear_grad()
    x1.clear_grad()
    y.backward()

    print(x0, x1, y)
    x0.data -= lr * x0.grad
    x1.data -= lr * x1.grad
