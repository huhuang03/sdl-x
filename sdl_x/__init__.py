from sdl_x.lose import mean_squared_error
from sdl_x.lose import cross_entropy_error

from sdl_x.gradient import numerical_gradient

from sdl_x.active import softmax
from sdl_x.active import sigmoid

from sdl_x.net import SimpleNet
from sdl_x.net import TwoLayerNet

from sdl_x.mnist import load_mnist

from sdl_x.layers import Affine
from sdl_x.layers import Relu
from sdl_x.layers import SoftmaxWithLoss

import numpy as np


def mean(src: np.ndarray, axis=None) -> np.ndarray:
    """
    求均值
    """
    if axis is None:
        axis = src