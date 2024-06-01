import numpy as np


def same_with_small_deviation(n1: np.ndarray, n2: np.ndarray, allowed_deviation=1e-7):
    """
    对比，并允许很小很小的误差
    """
    return n1.shape == n2.shape and np.all((n1 - n2) < allowed_deviation)

def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x