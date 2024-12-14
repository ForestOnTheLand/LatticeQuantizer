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


E6 = np.array([[0] * i + [-1, 1] + [0] * (6 - i) for i in range(1, 6)] + [[0.5] * 4 + [-0.5] * 4])
E6_dual = np.array(
    [[0] * i + [-1, 1] + [0] * (6 - i) for i in range(1, 5)]
    + [[0] + [2 / 3] * 2 + [-1 / 3] * 4 + [0]]
    + [[0.5] * 4 + [-0.5] * 4]
)
E7 = np.array([[0] * i + [-1, 1] + [0] * (6 - i) for i in range(6)] + [[0.5] * 4 + [-0.5] * 4])
E7_dual = np.array([[0] * i + [-1, 1] + [0] * (6 - i) for i in range(6)] + [[-0.75] * 2 + [0.25] * 6])
E8 = E8_dual = D_plus(8)

BEST_LATTICE: dict[int, NDArray] = {
    1: Z(1),
    2: A(2),
    3: A_dual(3),
    4: D(4),
    5: D_dual(5),
    6: E6_dual,
    7: E7_dual,
    8: E8,
    9: np.array(
        [
            [[2] + [0] * 8] + [[1] + [0] * i + [1] + [0] * (7 - i) for i in range(7)] + [[0.5] * 8 + [0.5732237949]],
        ]
    ),
    10: D_plus(10),
    11: np.array([[0] * (1 + i) + [1, -1] + [0] * (9 - i) for i in range(10)] + [[1 / 3] * 8 + [-2 / 3] * 4]),
    12: np.array(
        [
            [2] + [0] * 11,
            *[[1] + [0] * i + [1] + [0] * (10 - i) for i in range(5)],
            [0.5] * 6 + [1] + [0] * 5,
            *[[0] * 6 + [1] + [0] * i + [1] + [0] * (4 - i) for i in range(4)],
            [1] + [0] * 5 + [0.5] * 6,
        ]
    ),
    13: np.array(
        [[-1] + [0] * i + [1] + [0] * (11 - i) for i in range(7)]
        + [[0] * 8 + [2] + [0] * 4]
        + [[0] * 8 + [1] + [0] * i + [1] + [0] * (3 - i) for i in range(4)]
        + [[11 / 16] * 3 + [-5 / 16] * 5 + [1 / 2] * 5]
    ),
    14: (
        lambda q, a: np.array(
            [
                [2] + [0] * 13,
                [1, q] + [0] * 12,
                [0] * 2 + [2] + [0] * 11,
                [0] * 2 + [1, q] + [0] * 10,
                [0] * 4 + [2] + [0] * 9,
                [0] * 4 + [1, q] + [0] * 8,
                [1, 0, 0.5, -0.5 * q, 0.5, -0.5 * q, 1] + [0] * 7,
                [0.5, 0.5 * q, 1, 0, 1, 0, 0.5, 0.5 * q] + [0] * 6,
                [0.5, -0.5 * q, 1, 0, 0.5, -0.5 * q, 0, 0, 1] + [0] * 5,
                [1, 0, 0.5, 0.5 * q, 1, 0, 0, 0, 0.5, 0.5 * q] + [0] * 4,
                [1, 0, 1, 0, 0.5, 0.5 * q] + [0] * 4 + [a] + [0] * 3,
                [0] * 10 + [-a, a, 0, 0],
                [0] * 10 + [-a, 0, a, 0],
                [0.5, -0.5 * q, 0.5, -0.5 * q, 1] + [0] * 5 + [0.5 * a] * 4,
            ]
        )
    )(3**0.5, 25 / 19),
    15: np.array(
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
    ),
    16: np.array(
        [[4] + [0] * 15]
        + [[2] + [0] * i + [2] + [0] * (14 - i) for i in range(10)]
        + [[0] * i + [1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1] + [0] * (4 - i) for i in range(4)]
        + [[1] * 16]
    ),
}


def best_lattice(dim: int) -> NDArray:
    try:
        return BEST_LATTICE[dim].copy()
    except KeyError:
        raise NotImplementedError(f"not implemented for {dim=}")
