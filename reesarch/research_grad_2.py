import numpy as np

from sdl_x import Variable
from sdl_x import sin
from sdl_x import plot_dot_graph


x = Variable(np.array(0), name='x')
y = sin(x)
y.name = 'y'
plot_dot_graph(y, to_file='tmp_sin_forward.png')

y.backward()
gx = x.grad
gx.name = 'gx'
y.grad.name = 'gy'
# 到这里，已经错了呢
plot_dot_graph(gx, to_file='tmp_sin_backward.png')

x.clear_grad()
gx.backward()
print(x.grad)