from sdl_x.layer.liner import Liner
from sdl_x.model.model import Model


class TwoLayerNet(Model):
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.l1 = Liner(hidden_size)
        pass
        self.l2 = Liner(out_size)

    def forward(self, x):
        y = sigmoid(self.l1(x))
        y = self.l2(y)
        return y