from .function.function import Function
from .variable import Variable


def numerical_diff(f: Function, x: Variable, esp=1e-4):
    x0 = x - esp
    x1 = x + esp
    return (f(x1).data - f(x0).data) / 2 * esp
