# encoding=utf-8
# 我们使用数值微分的结果 来检验误差反向传播发的结果，理论上来说，结果应该相差非常小

import sys, os
import numpy as np
from sdl import load_mnist, TwoLayerNet

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)
x_batch = x_train[:3]
t_batch = t_train[:3]

print(t_batch)

grad_numerical = network.numerical_gradient(x_batch, t_batch)
grad_backprop = network.gradient(x_batch, t_batch)

# print(grad_numerical['b2'])
# print(grad_backprop['b2'])

# print(grad_numerical['W1'])
# print(np.sum(grad_numerical['W1']))
# print(grad_backprop['W1'])
# print(np.sum(grad_backprop['W1']))

for key in grad_numerical.keys():
    diff = np.average(np.abs(grad_backprop[key] - grad_numerical[key]))
    print(key + ": " + str(diff))

