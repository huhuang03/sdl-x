import numpy as np


def assert_same(n1: np.ndarray, n2: np.ndarray, allow_not_same=1e-5):
    diff = np.all(np.abs(n1 - n2) < allow_not_same)
    if not diff:
        raise ValueError(f"no equal\n: {n1}\n {n2}\n{n1 - n2}")
