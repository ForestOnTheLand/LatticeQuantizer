import numpy as np

from lattice.construct import construction
from lattice.nsm import numerical_nsm


if __name__ == "__main__":
    dim = 10
    A = construction(dim)
    with open(f"data/matrix_{dim}.npy", "wb") as file:
        np.save(file, A)
    with open(f"data/matrix_{dim}.npy", "rb") as file:
        A = np.load(file)
    error, var = numerical_nsm(A)
    print(f"Dimension {dim}. Estimated NSM: {error} with variance {var}")
