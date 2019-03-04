import numpy as np
from sdl.active import sigmoid, softmax
import matplotlib.pyplot as plt


def draw_sigmoid():
    x = np.arange(-8.0, 8.0, 0.1)
    y = sigmoid(x)
    plt.plot(x, y)
    plt.show()


draw_sigmoid()
