import numpy as np
from numpy.typing import NDArray


def Z(dim: int) -> NDArray:
    if not (isinstance(dim, int) and dim >= 1):
        raise ValueError(f"lattice Z(n) requires a positive integer n, given {dim}")
    return np.eye(dim, dim)


def A(dim: int) -> NDArray:
    if not (isinstance(dim, int) and dim >= 1):
        raise ValueError(f"lattice A(n) requires a positive integer n, given {dim}")
    G = np.eye(dim, dim + 1) - np.eye(dim, dim + 1, 1)
    return G


def A_dual(dim: int) -> NDArray:
    if not (isinstance(dim, int) and dim >= 1):
        raise ValueError(f"lattice A*(n) requires a positive integer n, given {dim}")
    G = np.zeros((dim, dim + 1))
    G[:-1, 0] = 1
    G[-1, 0] = -dim / (dim + 1)
    G[-1, 1:] = 1 / (dim + 1)
    G[range(dim - 1), range(1, dim)] = -1
    return G


def D(dim: int) -> NDArray:
    if not (isinstance(dim, int) and dim >= 3):
        raise ValueError(f"lattice D(n) requires a integer n >= 3, given {dim}")
    G = np.zeros((dim, dim))
    np.fill_diagonal(G, -1)
    G[0, 1] = -1
    G[range(1, dim), range(dim - 1)] = 1
    return G


def D_plus(dim: int) -> NDArray:
    if not (isinstance(dim, int) and dim > 3 and dim % 2 == 0):
        raise ValueError(f"lattice D+(n) requires an even integer n > 3, given {dim}")
    G = np.eye(dim, dim)
    G[0, 0] = 2
    G[range(1, dim - 1), range(dim - 2)] = -1
    G[-1, :] = 1 / 2
    return G


def D_dual(dim: int) -> NDArray:
    if not (isinstance(dim, int) and dim >= 3):
        raise ValueError(f"lattice D*(n) requires a integer n >= 3, given {dim}")
    G = np.eye(dim, dim)
    G[-1, :] = 1 / 2
    return G


def E(dim: int) -> NDArray:
    if dim == 6:
        return np.array(
            [
                [0.0, -1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, -1.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 1.0, 0.0],
                [0.5, 0.5, 0.5, 0.5, -0.5, -0.5, -0.5, -0.5],
            ]
        )
    elif dim == 7:
        return np.array(
            [
                [-1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, -1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, -1.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 1.0, 0.0],
                [0.5, 0.5, 0.5, 0.5, -0.5, -0.5, -0.5, -0.5],
            ]
        )
    elif dim == 8:
        return D_plus(8)
    else:
        raise ValueError(f"lattice E(n) requires n in (6, 7, 8), given {dim}")


def E_dual(dim: int) -> NDArray:
    if dim == 6:
        return np.array(
            [
                [0.0, -1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, -1.0, 1.0, 0.0, 0.0],
                [0.0, 2 / 3, 2 / 3, -1 / 3, -1 / 3, -1 / 3, -1 / 3, 0.0],
                [0.5, 0.5, 0.5, 0.5, -0.5, -0.5, -0.5, -0.5],
            ]
        )
    elif dim == 7:
        return np.array(
            [
                [-1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, -1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, -1.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 1.0, 0.0],
                [-0.75, -0.75, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25],
            ]
        )
    elif dim == 8:
        return D_plus(8)
    else:
        raise ValueError(f"lattice E*(n) requires n in (6, 7, 8), given {dim}")


def lambda_dual(dim: int) -> NDArray:
    if dim == 15:
        return np.array(
            [
                [2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
                [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
                [0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
                [0.5] * 15,
            ]
        )
    else:
        raise NotImplementedError(f"lattice Λ*(n) not implemented for n={dim}")


def best_lattice(dim: int) -> NDArray:
    if not (isinstance(dim, int) and dim > 0):
        raise ValueError(f"Expected dimension to be a positive integer, given {dim}")
    known_best = {
        1: Z,
        2: A,
        3: A_dual,
        4: D,
        5: D_dual,
        6: E_dual,
        7: E_dual,
        8: E,
        9: lambda _: np.array(
            [
                [2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5732237949],
            ]
        ),
        10: D_plus,
        15: lambda_dual,
    }
    try:
        return known_best[dim](dim)
    except KeyError:
        raise NotImplementedError(f"not implemented for {dim=}")
