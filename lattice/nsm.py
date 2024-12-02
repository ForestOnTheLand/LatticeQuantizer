import numpy as np
from numpy.typing import NDArray

import multiprocessing as mp

from lattice.core import clip


def numerical_error(B: NDArray, steps: int) -> tuple[float, float]:
    """
    Numerical quantization error by sampling.

    :param B: a generator matrix of the lattice
    :type B: NDArray
    :param steps: number of sampling
    :type steps: int
    :return: sum of error, sum of squared error (to compute variance)
    :rtype: tuple[float, float]
    """

    n, m = B.shape
    sum = 0.0
    sum_squared = 0.0
    for t in range(steps):
        z = np.random.standard_normal(n)
        u = clip(B, z @ B)
        y = z - u
        e = y @ B
        error = np.linalg.norm(e) ** 2
        sum += error
        sum_squared += error**2
    return sum, sum_squared


def numerical_nsm(B: NDArray, steps: int = 10000000, num_process: int = mp.cpu_count()) -> tuple[float, float]:
    """
    Estimate NSM numerically.

    :param B: a generator matrix of the lattice
    :type B: NDArray
    :param steps: number of sampling
    :type steps: int
    :return: estimated NSM
    :rtype: float
    """

    n, m = B.shape

    coeff = 1 / (n * np.linalg.det(B @ B.T) ** (1 / n))
    num_steps = [steps // num_process + (i < steps % num_process) for i in range(num_process)]
    with mp.Pool(num_process) as p:
        result = np.array(p.starmap(numerical_error, zip([B] * num_process, num_steps)))
    mean = result[:, 0].sum() / steps
    mean_squared = result[:, 1].sum() / steps
    var = mean_squared - mean**2

    return mean * coeff, var * coeff**2
