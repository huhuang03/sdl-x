import unittest
from .util import assert_same

import numpy
import numpy as np

from sdl_x import numerical_gradient


def x2_plus_x2(x):
    return np.sum(x ** 2)


class GradientTest(unittest.TestCase):

    @staticmethod
    def test_numerical_gradient():
        w = np.random.randn(28 * 28, 10)
        b = np.random.randn(1, 10)
        x = np.random.randn(1, 28 * 28) * 255
        loss = lambda p_w: np.sum(x @ p_w + b)
        # check can calc w gradient
        manually_dw = numerical_gradient(loss, w)
        math_dw = x.T @ numpy.ones((1, 10))
        assert_same(manually_dw, math_dw)
        print('here!!')

    # def test_gradient_decline(self):
    #     y =  gradient_decline(x2_plus_x2, np.array([5, 8]), 0.01)
    #     self.assertTrue(y < 1e-30)
    #     print(y)


if __name__ == '__main__':
    unittest.main()
