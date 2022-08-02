import pandas as pd
import time
import psutil
import itertools

from src.backends.density_matrix.compiler import DensityMatrixCompiler
from src.solvers.evolutionary_solver import EvolutionarySolver
from benchmarks.circuits import *
from src.metrics import Infidelity
from src.io import IO

from src.visualizers.solver_logs import plot_solver_logs


def run_combinations(solvers, targets, compilers, metrics):
    """
    Produces a benchmarking run for all possible combinations from lists of solvers, targets, compilers, and metrics.

    :param solvers: a list of Solver classes (not instanced)
    :type solvers: list
    :param targets: a list of the targets to
    :type targets: list
    :param compilers:
    :type compilers: list
    :param metrics:
    :type metrics: list
    :return: a list of dictionaries, each defining a sinle benchmarking run
    """

    c = itertools.product(solvers, targets, compilers, metrics)
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
            **solver_attr,
        )

        runs[f"run{i}"] = dict(
            solver=solver,
            target=target,  # TODO: as per Julie's suggestion, we could remove these as they're already in the solver
            compiler=compiler,
            metric=metric,
        )
    return runs


def benchmark_run(run: dict, name: str, io: IO):
    """
    Benchmarks one run, consisting of a unique combination of solver, target, compiler, and metric.

    :param run: dictionary containing the main class instances (solver, etc.)
    :type run: dict
    :param name: name of the run
    :type name: str
    :param io: IO object used to save data for the *current* run (not overall benchmarking IO)
    :type io: IO instance
    :return: dictionary storing all the results/metadata for the current run (can be added to a DataFrame)
    """
    solver = run["solver"]

    t0 = time.time()
    run["solver"].io = io
    run["solver"].save_openqasm = "both"
    run["solver"].solve()
    t1 = time.time()

    # here we can save specific information from each solver run, e.g., circuit figures, solver logs, etc.
    for log_name, log in solver.logs.items():
        io.save_dataframe(df=log, filename=f"log_{log_name}.csv")
    fig, axs = plot_solver_logs(solver.logs)
    io.save_figure(fig=fig, filename=f"log.pdf")

    # this line summarizes the performance of this solver run, is added to a pandas DataFrame
    d = dict(
        name=name,
        path=io.path.name,
        solver=run["solver"].__class__.__name__,
        compiler=run["compiler"].__class__.__name__,
        metric=run["compiler"].__class__.__name__,
        # target=run["target"],  # TODO: ideally the target is a QuantumState object, and we can use the __name__ here
        time=(t1 - t0),
        circuit_cost=solver.hof[0][0],
        circuit_depth=solver.hof[0][1].depth,
    )
    return d


def benchmark(runs: dict, io: IO, remote=True):
    """
    Runs a sequence of benchmark runs, either using parallel or serial processing.

    :param runs: a list of runs, each being a dictionary containing the important class instances (solvers, etc.)
    :type: list
    :param io: a top-level IO object for saving benchmark data
    :type io: IO instance
    :param remote: True/False flag. If True, will run solvers in parallel using `ray`
    :type remote: bool
    :return: DataFrame summarizing the results of all benchmarking runs
    """
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
            include_id=False,
            verbose=False,
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
        target=target, metric=metric, compiler=compiler, n_photon=3, n_emitter=1
    )
    example_run1 = {
        "solver": solver1,
        "target": target,
        "metric": metric,
        "compiler": compiler,
    }

    solver2 = EvolutionarySolver(
        target=target, metric=metric, compiler=compiler, n_photon=3, n_emitter=1
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
