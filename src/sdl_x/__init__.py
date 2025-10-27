from .active import sigmoid
from .active import softmax
from .function.add import add
from .function.mul import mul
from .function.neg import neg
from .function.pow import pow_
from .function.sub import sub, rsub
from .gradient import numerical_gradient
from .lose import cross_entropy_error
from .lose import mean_squared_error
from .mnist import load_mnist
from .net import SimpleNet, TwoLayerNet
from .v1.layer import Affine
from .v1.layer import Relu
from .v1.layer import SoftmaxWithLoss
from .variable import Variable
from .function.reshape import reshape
from .function.transpose import transpose
from .function.sum import sum_
from .square import Square

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