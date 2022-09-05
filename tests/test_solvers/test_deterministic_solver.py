import numpy as np
from src.backends.stabilizer.compiler import StabilizerCompiler
from src.backends.stabilizer.state import Stabilizer
from src.solvers.deterministic_solver import DeterministicSolver
from benchmarks.circuits import *
from src.metrics import Infidelity


def test_solver_initialization():
    compiler = StabilizerCompiler()
    _, target = linear_cluster_4qubit_circuit()
    metric = Infidelity(target)
    n_emitter = 1
    n_photon = 4
    solver = DeterministicSolver(
        target=target,
        metric=metric,
        compiler=compiler,
        n_emitter=n_emitter,
        n_photon=n_photon,
    )
    solver.solve()
    score, circuit = solver.hof
    print(score)
    circuit.draw_circuit()
