import numpy as np
import matplotlib.pyplot as plt

from lattice.construct import construction
from lattice.nsm import numerical_nsm
from lattice.visualize import close_points_plot


def run():
    dim = 10
    A = construction(dim, 100000000, 100, 5e-4, 1e3, progress_bar=False)
    with open(f"data/matrix_{dim}.npy", "wb") as file:
        np.save(file, A)
    # with open(f"data/matrix_{dim}.npy", "rb") as file:
    #     A = np.load(file)
    error, var = numerical_nsm(A, num_process=32)
    print(f"Dimension {dim}. Estimated NSM: {error} with variance {var}")


def visualize():
    # Lattice \Lambda_{15}^{*}
    A = np.array(
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
    x, y = close_points_plot(A)
    plt.plot(x, y)
    plt.yscale("log")
    plt.savefig(f"data/plot.png")


if __name__ == "__main__":
    run()
    visualize()
