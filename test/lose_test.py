import unittest
import numpy as np
from sdl_x import cross_entropy_error


class ActiveTest(unittest.TestCase):

    def test_cross_entropy_error(self):
        t = np.array([[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]])
        y = np.array([[0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]])

        # real is like 0.510825457099338
        self.assertTrue(cross_entropy_error(y, t) < 0.6)

        # real is like 2.3025840929945458
        y = np.array([[0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]])
        cross_entropy_error(np.array(y), t)
        self.assertTrue(cross_entropy_error(np.array(y), t) > 2)


if __name__ == '__main__':
    unittest.main()
