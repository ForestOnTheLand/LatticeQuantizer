import numpy as np
from numpy.typing import NDArray

from tqdm import tqdm

from lattice.core import reduce, clip


def orthogonalize(B: NDArray) -> NDArray:
    """
    Orthogonalization, based on Cholesky decomposing.

    :param B: the matrix to perform orthogonalization on, read-only
    :type B: NDArray
    :return: the decomposed matrix
    :rtype: NDArray
    """

    return np.linalg.cholesky(B @ B.T)


def construction(
    n: int,
    T: int = 1000000,
    Tr: int = 100,
    mu_0: float = 5e-3,
    nu: float = 2e2,
    delta: float = 0.75,
    checkpoint: NDArray | None = None,
    progress_bar: bool = True,
    seed: int | None = None,
) -> NDArray:
    """
    Main Procedure of Iterative Lattice Construction.
    Follows the Algorithm 1 in https://arxiv.org/abs/2401.01799, whose Table 1
    gives some recommended hyper-parameters.

    :param n: dimension
    :type n: int
    :param T: number of steps
    :type T: int
    :param Tr: reduction interval
    :type Tr: int
    :param mu_0: initial step size
    :type mu_0: float
    :param nu: ratio between initial and final step size
    :type nu: float
    :param delta: see `lattice.core.reduce`
    :type delta: float
    """

    rand = np.random.Generator(np.random.PCG64(seed))
    # random initialization or use checkpoint
    B = rand.standard_normal((n, n)) if checkpoint is None else checkpoint
    B = orthogonalize(reduce(B, delta))
    volume = np.prod(np.diag(B))
    B *= volume ** (-1 / n)

    for t in tqdm(range(T)) if progress_bar else range(T):
        # compute learning rate
        mu = mu_0 * nu ** (-t / (T - 1))
        # sample random vector
        z = rand.uniform(0, 1, n)
        u = clip(B, z @ B)
        y = z - u
        e = y @ B
        # gradient descent
        B -= mu * (np.tril(np.outer(y, e)) - np.diag(np.dot(e, e) / (n * np.diag(B))))
        # sanity check
        if np.any(np.diag(B) <= 0):
            raise RuntimeError(
                "Some diagonal element of the generator matrix is non-positive! "
                f"Parameters mu_0 ({mu_0}) and Tr ({Tr}) might be too large."
            )

        if t % Tr == Tr - 1:
            B = orthogonalize(reduce(B, delta))
            volume = np.prod(np.diag(B))
            B *= volume ** (-1 / n)

    return B


def numerical_grad(G: NDArray, steps: int):
    n, m = G.shape
    rand = np.random.Generator(np.random.PCG64())
    grad = np.zeros_like(G)
    for _ in tqdm(range(steps)):
        z = rand.uniform(0, 1, n)
        u = clip(G, z @ G)
        y = z - u
        e = y @ G
        grad += np.outer(y, e) - np.diag(np.dot(e, e) / (n * np.diag(G)))
    return grad / steps
