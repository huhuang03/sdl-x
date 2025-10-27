from .model import Model
from ..layer.liner import Liner


class MLP(Model):
    """
    Multi-Layer Perceptron (MLP) 多层感知器。MLP是全连接神经网络的别名
    """
    def __init__(self, fc_output_sizes, activation=sigmod):
        super().__init__()
        self.activation = activation
        self.layers = []

        for i, out_size in enumerate(fc_output_sizes):
            layer = Liner(out_size)
            setattr(self, 'l' + str(i), layer)
            self.layers.append(layer)

    def forward(self, x):
        for l in self.layers[:-1]:
            x = self.activation(l(x))
        return self.layers[-1](x)