import numpy as np
import matplotlib.pyplot as plt
import ray
from pathlib import Path
import json
import argparse

from lattice.construct import construction
from lattice.nsm import numerical_nsm
from lattice.visualize import close_points_plot

@ray.remote
def task(dimension: int, output_path: Path, plot_dir: Path, **config):
    A = construction(dimension, **config, progress_bar=False)
    error, var = numerical_nsm(A, num_process=16)
    print(f"Dimension {dimension}. Estimated NSM: {error} with variance {var}")
    with open(output_path, "a") as file:
        file.write(json.dumps({"dimension": dimension, "matrix": A.tolist() ,"error": error, "variance": var}) + "\n")
    x, y = close_points_plot(A)
    plt.plot(x, y)
    plt.yscale("log")
    plt.savefig(plot_dir / f"{dimension}.png")

if __name__ == "__main__":
    # Example usage: python run.py fast --min_dimension 10 --max_dimension 16
    ray.init()
    parser = argparse.ArgumentParser()
    parser.add_argument("config_name", type=str)
    parser.add_argument("--min_dimension", type=int, default=10)
    parser.add_argument("--max_dimension", type=int, default=16)
    args = parser.parse_args()
    config_name = args.config_name
    config = json.load(open(f"configs/{config_name}.json"))
    output_path = Path(f"results/{config_name}.jsonl")
    plot_dir = Path(f"plots/{config_name}")
    plot_dir.mkdir(parents=True, exist_ok=True)
    candidate_dimensions = range(args.min_dimension, args.max_dimension + 1)
    # skip all dimensions that have been computed
    if output_path.exists():
        with open(output_path, "r") as file:
            computed_dimensions = set(json.loads(line)["dimension"] for line in file)
        dimensions = [dimension for dimension in candidate_dimensions if dimension not in computed_dimensions]
    else:
        dimensions = candidate_dimensions
    print(f"{len(candidate_dimensions)-len(dimensions)} dimensions have been computed, {len(dimensions)} dimensions to go.")
    tasks = [task.remote(dimension, output_path, plot_dir, **config) for dimension in dimensions]
    ray.get(tasks)