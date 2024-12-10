import numpy as np
from numpy.typing import NDArray

import multiprocessing as mp
from tqdm import tqdm

from lattice.core import clip


def numerical_error(B: NDArray, steps: int, progress_bar: bool = False) -> tuple[float, float]:
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
    rand = np.random.Generator(np.random.PCG64())
    for t in tqdm(range(steps)) if progress_bar else range(steps):
        z = rand.uniform(0, 1, n)
        u = clip(B, z @ B)
        y = z - u
        e = y @ B
        error = np.linalg.norm(e) ** 2
        sum += error
        sum_squared += error**2
    return sum, sum_squared


def numerical_nsm(B: NDArray, steps: int = 10000000, num_process: int = 1) -> tuple[float, float]:
    """
    Estimate NSM numerically.

    :param B: a generator matrix of the lattice
    :type B: NDArray
    :param steps: number of sampling
    :type steps: int
    :return: estimated NSM, with the sampling variance
    :rtype: tuple[float, float]
    """

    n, m = B.shape

    coeff = 1 / (n * np.linalg.det(B @ B.T) ** (1 / n))
    num_steps = [steps // num_process + (i < steps % num_process) for i in range(num_process)]
    if num_process > 1:
        with mp.Pool(num_process) as p:
            result = np.array(p.starmap(numerical_error, zip([B] * num_process, num_steps)))
        sum_error, sum_squared_error = result.sum(axis=0)
    else:
        sum_error, sum_squared_error = numerical_error(B, steps, progress_bar=True)
    mean = sum_error / steps
    mean_squared = sum_squared_error / steps
    var = (mean_squared - mean**2) / (steps - 1)

    return mean * coeff, var**0.5 * coeff
