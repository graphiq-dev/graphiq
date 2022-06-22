import pandas as pd
import time

from src.backends.density_matrix.compiler import DensityMatrixCompiler
from src.solvers.rule_based_random_solver import RuleBasedRandomSearchSolver
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


def run_solver(_io, target, Solver, Metric, Compiler, solver_attrs, target_attrs):
    metric = Metric(target=target)
    compiler = Compiler()

    solver = Solver(target=target, metric=metric, compiler=compiler,
                    n_photon=target_attrs['n_photon'], n_emitter=target_attrs['n_emitter'])
    update(solver, solver_attrs)

    t0 = time.time()
    solver.solve()
    t1 = time.time()

    # here we can save specific information from each solver run, e.g., circuit figures, solver logs, etc.
    _io.save_txt("nothing from this solver run has been saved", "run.txt")

    # this line summarizes the performance of this solver run, is added to a pandas DataFrame
    d = dict(
        solver=Solver.__name__,
        compiler=Compiler.__name__,
        metric=Metric.__name__,
        target=target_attrs['name'],
        time=(t1-t0),
        min_cost=solver.hof[0][0],
        path=io.path,
    )
    return d


if __name__ == "__main__":
    #%% provide all combinations of solvers, targets, compilers, and metrics to run
    solvers = [
        (RuleBasedRandomSearchSolver, dict(n_stop=100, n_pop=50, n_hof=5, tournament_k=2)),
        (RuleBasedRandomSearchSolver, dict(n_stop=100, n_pop=50, n_hof=5, tournament_k=5)),
    ]

    # provide a list of targets
    targets = [
        (ghz3_state_circuit()[1]['dm'], dict(name='ghz3', n_photon=3, n_emitter=1)),
        (ghz4_state_circuit()[1]['dm'], dict(name='ghz4', n_photon=4, n_emitter=1)),
    ]

    compilers = [
        DensityMatrixCompiler
    ]

    metrics = [
        Infidelity,
    ]

    #%%
    io = IO.new_directory(folder="benchmarks", include_date=True, include_time=True, include_id=False)
    remote = True
    if remote:
        import ray
        ray.init(ignore_reinit_error=True)
        run_solver_remote = ray.remote(run_solver)

    futures = []

    for Compiler in compilers:
        for (target, target_attrs) in targets:
            for Metric in metrics:
                for (Solver, solver_attrs) in solvers:
                    io_tmp = IO.new_directory(path=io.path, folder=" test ",
                                              include_date=True, include_time=True, include_id=True)

                    if remote:
                        futures.append(
                            run_solver_remote.remote(io_tmp, target, Solver, Metric, Compiler, solver_attrs, target_attrs)
                        )
                    else:
                        futures.append(
                            run_solver(io_tmp, target, Solver, Metric, Compiler, solver_attrs, target_attrs)
                        )

    if remote:
        futures = ray.get(futures)

    df = pd.DataFrame(futures)
    io.save_dataframe(df, "benchmark_solver_run.csv")
