import unittest

import numpy as np
from sdl import numerical_gradient
from sdl.gradient import gradient_decline


def x2_plus_x2(x):
    return np.sum(x ** 2)


class ActiveTest(unittest.TestCase):

    def test_numerical_gradient(self):
        gradient = numerical_gradient(x2_plus_x2, np.array([3.0, 4.0]))
        # should be [6, 8]
        print(gradient)

    def test_gradient_decline(self):
        y =  gradient_decline(x2_plus_x2, np.array([5, 8]), 0.01)
        self.assertTrue(y < 1e-30)
        print(y)


if __name__ == '__main__':
    unittest.main()
