import pandas as pd
import time
import psutil
import itertools

from src.backends.density_matrix.compiler import DensityMatrixCompiler
from src.solvers.evolutionary_solver import EvolutionarySolver
from benchmarks.circuits import *
from src.metrics import Infidelity
from src.io import IO

from benchmarks.pipeline import benchmark, run_combinations


if __name__ == "__main__":
    # %% provide all combinations of solvers, targets, compilers, and metrics to run as a list
    solvers = [
        (EvolutionarySolver, dict(n_stop=30, n_pop=80, n_hof=10, tournament_k=2)),
        (EvolutionarySolver, dict(n_stop=30, n_pop=100, n_hof=10, tournament_k=2)),
    ]

    # provide a list of targets
    targets = [
        (ghz3_state_circuit()[1]["dm"], dict(name="ghz3", n_photon=3, n_emitter=1)),
        (ghz4_state_circuit()[1]["dm"], dict(name="ghz4", n_photon=4, n_emitter=1)),
    ]

    compilers = [DensityMatrixCompiler]

    metrics = [Infidelity]

    # take all combinations of the lists provided above
    runs = run_combinations(solvers, targets, compilers, metrics)

    io = IO.new_directory(
        folder="benchmarks", include_date=True, include_time=True, include_id=False
    )
    df = benchmark(runs=runs, io=io, remote=True)
    print(df)
