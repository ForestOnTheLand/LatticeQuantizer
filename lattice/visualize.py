import numpy as np
from numpy.typing import NDArray

import matplotlib.pyplot as plt


def close_points(G: NDArray, r: float) -> list[NDArray]:
    """
    Find the lattice points that are close to the origin (i.e. the distance is less/equal than norm).

    :param G: the generator matrix of lattice
    :type G: NDArray
    :param r: the maximum distance
    :type r: float
    :return: the list of vectors, each vector `x` represent the lattice point `x @ G`
    :rtype: list[NDArray]
    """

    def sign(x: int | float) -> int:
        return 1 if x > 0 else -1

    n, m = G.shape
    H = np.linalg.inv(G)
    assert m == n

    dist = np.zeros(n)
    e = np.zeros((n, n))
    u = np.zeros(n, dtype=np.int32)
    step = np.zeros(n, dtype=np.int32)

    U: list[NDArray] = []

    k = n - 1
    eps = 1e-6
    norm = r**2

    u[k] = round(e[k, k])
    y = (e[k, k] - u[k]) / H[k, k]
    step[k] = sign(y)

    while True:
        newdist = dist[k] + y**2
        if newdist < (1 + eps) * norm:
            if k != 0:
                for i in range(k):
                    e[k - 1, i] = e[k, i] - y * H[k, i]
                k -= 1
                dist[k] = newdist
                u[k] = round(e[k, k])
                y = (e[k, k] - u[k]) / H[k, k]
                step[k] = sign(y)
            else:
                U.append(u.copy())
                u[k] += step[k]
                y = (e[k, k] - u[k]) / H[k, k]
                step[k] = -step[k] - sign(step[k])
        else:
            if k == n - 1:
                return U
            else:
                k += 1
                u[k] += step[k]
                y = (e[k, k] - u[k]) / H[k, k]
                step[k] = -step[k] - sign(step[k])


def close_points_plot(G: NDArray, max_r: float = 2.5, num: int = 1001) -> tuple[NDArray, NDArray]:
    """
    Plot the cumulative distribution of lattice point norms.

    :param G: the generator matrix of lattice
    :type G: NDArray
    :param max_r: the maximum distance
    :type max_r: float
    :param num: the number of points sampled on the x-axis (more points, more accurate)
    :type num: int
    :return: x_space and y_space used in plt.plot
    :rtype: tuple[NDArray, NDArray]
    """

    n, m = G.shape
    G *= np.linalg.det(G) ** (-1 / n)
    U = close_points(G, max_r)
    distance = np.array([np.linalg.norm(u @ G) ** 2 for u in U])
    x_space = np.linspace(0, max_r**2, num)
    y_space = np.array([np.sum(distance < x) for x in x_space])
    return x_space, y_space
