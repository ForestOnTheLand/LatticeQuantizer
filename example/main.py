import numpy as np
import matplotlib.pyplot as plt
import argparse

from lattice import construction, orthogonalize, numerical_nsm, close_points_plot, classical


def run(dim: int):
    # A = construction(dim)
    A = construction(dim, T=10000000, mu_0=1e-3, nu=5e2)
    # A = construction(dim, T=100000000, mu_0=5e-4, nu=1e3)

    with open(f"data/matrix_{dim}.npy", "wb") as file:
        np.save(file, A)
    error, var = numerical_nsm(A, num_process=32)
    print(f"Dimension {dim}. Estimated NSM: {error} with variance {var}")

    x, y = close_points_plot(A)
    plt.plot(x, y, label="numerical")
    plt.yscale("log")


def visualize(dim: int):
    try:
        A = orthogonalize(classical.best_lattice(dim))
        x, y = close_points_plot(A)
        plt.plot(x, y, label="optimal")
        plt.yscale("log")
    except NotImplementedError:
        print(f"Unknown Best Lattice in dimension {dim}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", "-n", type=int, dest="dim")
    args = parser.parse_args()

    dim = args.dim
    run(dim)
    visualize(dim)
    plt.xlabel("$r^2$")
    plt.ylabel("count")
    plt.legend()
    plt.savefig(f"data/plot_{dim}.png")
