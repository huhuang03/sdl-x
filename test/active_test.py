import unittest
import numpy as np
from sdl_x.active import sigmoid
from sdl_x.active import softmax


class ActiveTest(unittest.TestCase):

    def test_sigmoid(self):
        a = np.array([[0]])
        self.assertTrue(sigmoid(a) == np.array([1/2]))
        a = np.array([[0], [0]])
        self.assertTrue(a.shape == (2, 1))

    def test_relu(self):
        a = np.array([[0, 1], [-1, 2]])
        self.assertTrue(np.array_equal(np.maximum(0, a), np.array([[0, 1], [0, 2]])))

    def test_softmax(self):
        a = np.array([1, 1, 1])
        self.assertTrue(np.array_equal(softmax(a), np.array([1/3.0, 1/3.0, 1/3.0])))
        a = np.array([[1, 1, 1], [1, 1, 1]])
        self.assertTrue(np.array_equal(softmax(a), np.array([[1/3.0, 1/3.0, 1/3.0], [1/3.0, 1/3.0, 1/3.0]])))


if __name__ == '__main__':
    unittest.main()
