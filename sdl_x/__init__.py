from sdl_x.active import sigmoid
from sdl_x.active import softmax
from sdl_x.function.add import add
from sdl_x.function.mul import mul
from sdl_x.function.neg import neg
from sdl_x.function.pow import pow_
from sdl_x.function.sub import sub, rsub
from sdl_x.gradient import numerical_gradient
from sdl_x.lose import cross_entropy_error
from sdl_x.lose import mean_squared_error
from sdl_x.mnist import load_mnist
from sdl_x.net import SimpleNet, TwoLayerNet
from sdl_x.v1.layer import Affine
from sdl_x.v1.layer import Relu
from sdl_x.v1.layer import SoftmaxWithLoss
from .variable import Variable
from .function.reshape import reshape
from .function.transpose import transpose
from .function.sum import sum_

Variable.__mul__ = mul
Variable.__rmul__ = mul
Variable.__add__ = add
Variable.__radd__ = add
Variable.__pow__ = pow_
Variable.__sub__ = sub
Variable.__rsub__ = rsub
Variable.__neg__ = neg
Variable.real_reshape = reshape
Variable.transpose = transpose
Variable.real_sum = sum_