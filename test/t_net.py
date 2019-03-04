import unittest

import numpy as np

from sdl import load_mnist, TwoLayerNet


class NetTest(unittest.TestCase):

    def test_two_layer_net(self):
        (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label = True)
        network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)
        t_size = 2
        x_batch = x_train[:t_size]
        t_batch = t_train[:t_size]

        grad_numerical= network.numerical_gradient(x_batch, t_batch)
        grad_backprop = network.gradient(x_batch, t_batch)
        # 求各个权重的绝对误差的平均值
        for key in grad_numerical.keys():
            diff = np.average(np.abs(grad_backprop[key] - grad_numerical[key]))
            print(key + ":" + str(diff))


if __name__ == '__main__':
    unittest.main()
