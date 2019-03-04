import numpy as np
from mpl_toolkits.mplot3d import axes3d, Axes3D
import matplotlib.pyplot as plt

def fun1(x, y):
    """z = 1 / 20 * x ^ 2 + y ^ 2"""
    return 1 / 20 * np.square(x) + np.square(y)


def draw_sgd_bad_fun():
    """
    draw z = 1 / 20 * x ^ 2 + y ^ 2
    """
    # z = fun1(x, y)
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    item = np.arange(-10, 10, 0.2)
    for i in item:
        x = np.repeat(i, 200)
        y = item
        z = fun1(x, y)
        # x = np.repeat(item, 200)
        ax.plot(x, y, z, '-b')
    plt.show()
    # for i in np.arange(-10, 10, 0.1):
    #     x = i
    #     y =
    #     for j in np.array(-10, 10, 1):


draw_sgd_bad_fun()
