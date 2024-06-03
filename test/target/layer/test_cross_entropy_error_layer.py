from sdl_x.v1.layer.loss import CrossEntropyError
import numpy as np


# noinspection PyMethodMayBeStatic
class TestCrossEntropyError:
    def test_t_compatible(self):
        # 3 x 2
        y = np.array([[1, 0, 0], [0, 1, 0]])
        t = np.array([1, 0])
        loss_fn = CrossEntropyError()
        loss = loss_fn.forward(y, t)