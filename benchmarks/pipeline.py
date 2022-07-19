import pandas as pd
import time
import psutil
import itertools

from src.backends.density_matrix.compiler import DensityMatrixCompiler
from src.solvers.evolutionary_solver import EvolutionarySolver
from benchmarks.circuits import *
from src.metrics import Infidelity
from src.io import IO


def update(obj, attrs):
    """
    Updates object attributes from a dictionary. Used to update solver attributes, e.g., n_stop, n_pop
    :param obj:
    :param attrs:
    :return:
    """
    for key, value in attrs.items():
        setattr(obj, key, value)
    return


def run_combinations(solvers, targets, compilers, metrics):
    c = list(itertools.product(solvers, targets, compilers, metrics))
    runs = {}
    for (i, c) in enumerate(c):
        ((Solver, solver_attr), (target, target_attr), Compiler, Metric) = c

        metric = Metric(target=target)
        compiler = Compiler()

        solver = Solver(
            target=target,
            metric=metric,
            compiler=compiler,
            n_photon=target_attr["n_photon"],
            n_emitter=target_attr["n_emitter"],
        )
        update(solver, solver_attr)

        runs[f"run{i}"] = dict(
            solver=solver,
            target=target,
            compiler=compiler,
            metric=metric,
        )
    return runs


def benchmark_run(run: dict, name: str, io: IO):
    """

    :param run:
    :param name:
    :param io:
    :return:
    """
    solver = run["solver"]

    t0 = time.time()
    run["solver"].solve()
    t1 = time.time()

    # here we can save specific information from each solver run, e.g., circuit figures, solver logs, etc.
    io.save_txt("nothing from this solver run has been saved", "run.txt")

    # this line summarizes the performance of this solver run, is added to a pandas DataFrame
    d = dict(
        solver=run["solver"].__class__.__name__,
        compiler=run["compiler"].__class__.__name__,
        metric=run["compiler"].__class__.__name__,
        target=run["target"],
        name=run["solver"].__class__.__name__,
        time=(t1 - t0),
        min_cost=solver.hof[0][0],
    )
    return d


def benchmark(runs: dict, io: IO, remote=True):

    if remote:
        import ray

        ray.init(num_cpus=psutil.cpu_count() - 1, ignore_reinit_error=True)
        benchmark_run_remote = ray.remote(benchmark_run)

    futures = []

    t0 = time.time()
    for name, run in runs.items():
        io_tmp = IO.new_directory(
            path=io.path,
            folder=name,
            include_date=True,
            include_time=True,
            include_id=True,
        )

        if remote:
            futures.append(benchmark_run_remote.remote(run, name=name, io=io_tmp))
        else:
            futures.append(benchmark_run(run, name=name, io=io_tmp))

    if remote:
        futures = ray.get(futures)

    df = pd.DataFrame(futures)
    io.save_dataframe(df, "benchmark_solver_run.csv")
    print(f"Total time {time.time() - t0}")
    return df


if __name__ == "__main__":
    # define a single run by creating an instance of each object
    target = ghz3_state_circuit()[1]["dm"]
    compiler = DensityMatrixCompiler()
    metric = Infidelity(target=target)

    solver1 = EvolutionarySolver(
        target=target, metric=metric, compiler=compiler, n_photon=3, n_emitter=1, selection_active=False
    )
    example_run1 = {
        "solver": solver1,
        "target": target,
        "metric": metric,
        "compiler": compiler,
    }

    solver2 = EvolutionarySolver(
        target=target,
        metric=metric,
        compiler=compiler,
        n_photon=3,
        n_emitter=1,
        selection_active=False,
    )

    example_run2 = {
        "solver": solver2,
        "target": target,
        "metric": metric,
        "compiler": compiler,
    }

    # put all runs into a dict, with a descriptive name
    runs = {
        "example_run1": example_run1,
        "example_run2": example_run2,
    }

    io = IO.new_directory(
        folder="benchmarks", include_date=True, include_time=True, include_id=False
    )
    df = benchmark(runs=runs, io=io, remote=True)
    print(df)
