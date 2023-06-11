import unittest
from .util import assert_same

import numpy
import numpy as np

from sdl_x import numerical_gradient


def x2_plus_x2(x):
    return np.sum(x ** 2)


class GradientTest(unittest.TestCase):

    def test_numerical_gradient(self):
        w = np.random.randn(28 * 28, 10)
        b = np.random.randn(1, 10)
        x = np.random.randn(1, 28 * 28) * 255
        loss = lambda p_w: np.sum(x @ p_w + b)
        # check can calc w gradient
        manually_dw = numerical_gradient(loss, w)
        math_dw = x.T @ numpy.ones((1, 10))
        assert_same(manually_dw, math_dw)
        # calc by manually

        # gradient = numerical_gradient(loss, w)
        # print(gradient)
    # def test_numerical_gradient_dim_2(self):
    #     f = lambda x: 2 * math.pow(x[0], 2) + 3 * math.pow(x[1], 2)
    #     gradient = numerical_gradient(f, np.array([3.0, 4.0]))
    #     assert self._check_equal_allow_small_diff(gradient, np.array([12., 24.]))
    #
    # def _check_equal_allow_small_diff(self, arr1, arr2):
    #     allow_diff = 1e-5
    #     return np.all((arr1 - arr2) < allow_diff)
    #
    # def test_numerical_gradient(self):
    #     # should not be like this.
    #     gradient = numerical_gradient(x2_plus_x2, np.array([[3.0], [4.0]]))
    #     self._check_equal_allow_small_diff(gradient, np.array([[6.], [8.]]))
    #
    # def test_gradient_decline(self):
    #     y =  gradient_decline(x2_plus_x2, np.array([5, 8]), 0.01)
    #     self.assertTrue(y < 1e-30)
    #     print(y)


if __name__ == '__main__':
    unittest.main()
