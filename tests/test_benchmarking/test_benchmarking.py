from graphiq.backends.density_matrix.compiler import DensityMatrixCompiler
from graphiq.solvers.evolutionary_solver import EvolutionarySolver
from benchmarks.circuits import *
from graphiq.metrics import Infidelity
from graphiq.io import IO

from benchmarks.pipeline import benchmark
from graphiq.state import QuantumState


def test_one_benchmarking_run():
    # define a single run by creating an instance of each object
    _, target = ghz3_state_circuit()
    compiler = DensityMatrixCompiler()
    metric = Infidelity(target=target)

    solver1 = EvolutionarySolver(
        target=target, metric=metric, compiler=compiler, n_photon=3, n_emitter=1
    )
    solver1.n_stop = 10
    solver1.n_pop = 5

    example_run1 = {
        "solver": solver1,
        "target": target,
        "metric": metric,
        "compiler": compiler,
    }

    # put all runs into a dict, with a descriptive name
    runs = {
        "example_run1": example_run1,
    }

    io = IO.new_directory(
        folder="test_benchmarks",
        include_date=False,
        include_time=False,
        include_id=False,
    )
    # df = benchmark(runs=runs, io=io, remote=False)
    # print(df)


if __name__ == "__main__":
    test_one_benchmarking_run()
