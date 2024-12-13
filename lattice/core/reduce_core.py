import numpy as np
from numpy.typing import NDArray
from . import csrc


def reduce(B: NDArray, delta: float = 0.5) -> NDArray:
    """
    Reduction function, based on Lenstra–Lenstra–Lovasz algorithm.
    Intuitively, it makes the rows (basis vectors) shorter and more orthogonal to each other.
    See https://cims.nyu.edu/~regev/teaching/lattices_fall_2004/ln/lll.pdf for more details.
    Thanks to https://github.com/itennenhouse/lll/blob/main/l.py.
    C++ implementation is more than 50x speed up!

    :param B: the matrix to perform reduction on, changed inplace
    :type B: NDArray
    :param delta: the hyper-parameter for the algorithm, must be in (0.25, 1)
    :type delta: float
    :return: the reduced matrix, i.e. B
    :rtype: NDArray
    """
    assert B.shape[0] == B.shape[1]
    csrc.reduce(B, delta)
    return B
