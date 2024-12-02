import numpy as np
from numpy.typing import NDArray
import time
from lattice.core import reduce


def gram_schmidt(B: NDArray) -> NDArray:
    """
    Gram-Schmidt Orthogonalization without normalize.

    :param B: the matrix to perform Gram-Schmidt on, read-only
    :type B: NDArray
    :return: the result
    :rtype: NDArray
    """

    n, m = B.shape
    B = B.copy()
    for i in range(1, n):
        B[i] = B[i] - np.add.reduce([np.dot(B[i], b) / (np.dot(b, b)) * b for b in B[:i]])
    return B


def reduce_std(B: NDArray, delta: float = 0.5) -> NDArray:
    """
    Reduction function, based on Lenstra–Lenstra–Lovasz algorithm.
    Intuitively, it makes the rows (basis vectors) shorter and more orthogonal to each other.
    See https://cims.nyu.edu/~regev/teaching/lattices_fall_2004/ln/lll.pdf for more details.
    Thanks to https://github.com/itennenhouse/lll/blob/main/l.py.
    Standard implementation using python.

    :param B: the matrix to perform reduction on, changed inplace
    :type B: NDArray
    :param delta: the hyper-parameter for the algorithm, must be in (0.25, 1)
    :type delta: float
    :return: the reduced matrix, i.e. B
    :rtype: NDArray
    """

    if not 0.25 < delta < 1:
        raise ValueError(f"Delta must be in range (0.25, 1)! Given: {delta}.")

    n, m = B.shape
    while True:
        Q = gram_schmidt(B)
        # reduction step
        for i in range(1, n):
            for j in range(i - 1, -1, -1):
                B[i] -= round(np.dot(B[i], Q[j]) / np.dot(Q[j], Q[j])) * B[j]
        # swap step
        for i in range(0, n - 1):
            # check if there is elements that violate the check
            v = np.dot(B[i + 1], Q[i]) / np.dot(Q[i], Q[i]) * Q[i] + Q[i + 1]
            if delta * np.linalg.norm(Q[i]) > np.linalg.norm(v):
                B[[i, i + 1]] = B[[i + 1, i]]
                break
        else:
            # no elements violate the check, we're done.
            return B


if __name__ == "__main__":
    print("Testing correctness: ")
    for _ in range(10):
        A = np.random.standard_normal((10, 10))
        B_std = reduce_std(A.copy())
        B_out = reduce(A.copy())
        np.testing.assert_allclose(B_out, B_std)
    print("Correctness passed!")
    print(f"Testing Speed:")
    ITER = 5000
    start = time.time()
    for _ in range(ITER):
        A = np.random.standard_normal((10, 10))
        reduce_std(A)
    cost = time.time() - start
    print(f"Python: {ITER} iter / {cost} sec = {ITER/cost} iteration per second in average.")
    ITER = 100000
    start = time.time()
    for _ in range(ITER):
        A = np.random.standard_normal((10, 10))
        reduce(A)
    cost = time.time() - start
    print(f"C++: {ITER} iter / {cost} sec = {ITER/cost} iteration per second in average.")
