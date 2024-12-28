# Term Project of Machine Learning Theory, 2024 fall

## Environment

The code are tested under Python 3.12. A minimum version of Python 3.10 is required. The dependencies and c++ extension can be installed by running

```bash
pip install -e .
```

## File Structure

- `./example`: Example usage.
- `./test`: Correctness for C++ code (compared with Python implementation).
- `./lattice`: Implementation code, in Python.
- `./csrc`: Helping code, in C++.
- `./experiment`: Code for running experiments and experiment results.
    - `final_plots`: Plots from dimension 10 to 20.
    - `final_result.jsonl`: Generator matrices and NSM from dimension 10 to 20. Each line is a dictionary with keys "dimension", "A", "error", "var".

## Running experiments

To reproduce our results, you can run
```bash
cd experiment
python run.py --dimensions 10 11 12 13 14 15 16 17 [--seed] [--name]
```
This will take ~5 hours on a multi-core server (ours has 96 CPUs). Your results and plots will be stored under directory `plots` and `results` with given name (default is date). Any dimensions already computed with the same name will be skipped. You should get the same results as ours with default seed 42. (For dimension 15, use seed 2025 instead!)

You can also try higher dimensions by running
```bash
python run.py --dimensions 18 19 20
```
This will take **3 days or more** as we increase running timesteps to get it converge. And it's normal if it doesn't converge to our results because we ran several times to find a best result in high dimensions.