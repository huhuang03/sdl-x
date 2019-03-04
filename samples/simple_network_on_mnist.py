import pickle

import numpy as np
from sdl.active import sigmoid, softmax
from sdl.mnist import load_mnist


def get_test_data():
    (x_test, t_test), = load_mnist()[1:]
    return x_test, t_test


def init_network():
    with open('../doc/asset/sample_weight.pkl', 'rb') as f:
        return pickle.load(f)


def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)
    return y


x, t = get_test_data()
network = init_network()
y = predict(network, x)

size = len(x)
p = np.argmax(y, axis=1)
accuracy_cnt = np.sum(p == t)


print("Accuracy: {}".format(float(accuracy_cnt) / size))


