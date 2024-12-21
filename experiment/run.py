import numpy as np
import matplotlib.pyplot as plt
import ray
from pathlib import Path
import json
import argparse

from lattice import classical
from lattice.construct import construction, orthogonalize
from lattice.nsm import numerical_nsm
from lattice.visualize import close_points_plot
from datetime import datetime

NUM_TRIALS = 20

def visualize(dimension: int, A, plot_dir: Path):
    plt.figure()
    try:
        A_star = orthogonalize(classical.best_lattice(dimension))
        x, y = close_points_plot(A_star)
        plt.plot(x, y, label="optimal")
        plt.yscale("log")
    except NotImplementedError:
        print(f"Unknown Best Lattice in dimension {dimension}.")

    x, y = close_points_plot(A)
    plt.plot(x, y, label="numerical")
    plt.yscale("log")
    plt.legend()
    plt.savefig(plot_dir / f"{dimension}.png")

@ray.remote
def task(dimension: int, config_name: str):
    config = json.load(open(f"configs/{config_name}.json"))
    A = construction(dimension, **config, progress_bar=False)
    error, var = numerical_nsm(A, num_process=4)
    return A, error, var

@ray.remote
def search(dimension: int, output_path: Path, plot_dir: Path):
    '''
    Search for the best lattice in the given dimension.
    First run 'medium' config for 20 times to find a good starting point.
    Then use the matrix with lowest NSM as checking point to run 'slow' config.
    '''
    tasks = [task.remote(dimension, "medium") for _ in range(NUM_TRIALS)]
    results = ray.get(tasks)
    best_A = min(results, key=lambda x: x[1])[0]
    print(f"Dimension {dimension}. Best matrix from medium config found.")
    A = construction(dimension, **json.load(open("configs/slow.json")), progress_bar=False, checkpoint=best_A)
    error, var = numerical_nsm(A, num_process=8)
    print(f"Dimension {dimension}. Estimated NSM: {error} with variance {var}")
    with open(output_path, "a") as file:
        file.write(json.dumps({"dimension": dimension, "matrix": A.tolist() ,"error": error, "variance": var}) + "\n")
    visualize(dimension, A, plot_dir)

if __name__ == "__main__":
    # Example usage: python run.py --dimensions 10 11 12 13 14 15 16
    ray.init()
    parser = argparse.ArgumentParser()
    parser.add_argument("--dimensions", type=int, nargs='+', help="List of dimensions to run", default=[10, 11, 12, 13, 14, 15, 16])
    args = parser.parse_args()
    timestamp = datetime.now().strftime("%Y%m%d")
    output_path = Path(f"results/{timestamp}.jsonl")
    plot_dir = Path(f"plots/{timestamp}")
    plot_dir.mkdir(parents=True, exist_ok=True)
    candidate_dimensions = args.dimensions
    # skip all dimensions that have been computed
    if output_path.exists():
        with open(output_path, "r") as file:
            computed_dimensions = set(json.loads(line)["dimension"] for line in file)
        dimensions = [dimension for dimension in candidate_dimensions if dimension not in computed_dimensions]
    else:
        dimensions = candidate_dimensions
    print(f"{len(candidate_dimensions)-len(dimensions)} dimensions have been computed, {len(dimensions)} dimensions to go.")
    tasks = [search.remote(dimension, output_path, plot_dir) for dimension in dimensions]
    ray.get(tasks)