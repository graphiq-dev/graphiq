import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import time
from benchmarks.graph_states import *
from src.backends.stabilizer.compiler import StabilizerCompiler
from src.backends.stabilizer.functions.rep_conversion import (
    get_clifford_tableau_from_graph,
)
from src.solvers.evolutionary_solver import EvolutionarySearchSolverSetting
from src.solvers.hybrid_solvers import HybridEvolutionarySolver
from benchmarks.circuits import *
from src.metrics import Infidelity
from src.state import QuantumState
from benchmarks.alternate_circuits import *
import src.noise.noise_models as noise


def deterministic_solver_runtime(n_low, n_high, n_step):
    compiler_runtime = []
    solver_runtime = []
    for n_inner_photons in range(n_low, n_high, n_step):
        target_tableau = get_clifford_tableau_from_graph(
            repeater_graph_states(n_inner_photons)
        )
        n_photon = target_tableau.n_qubits
        target = QuantumState(n_photon, target_tableau, representation="stabilizer")
        compiler = StabilizerCompiler()
        metric = Infidelity(target)

        solver = DeterministicSolver(
            target=target,
            metric=metric,
            compiler=compiler,
        )
        start_time = time.time()
        solver.solve()
        solver_duration = time.time() - start_time
        score, circuit = solver.result
        assert np.allclose(score, 0.0)
        solver_runtime.append(solver_duration)
        start_time = time.time()
        compiler.compile(circuit)
        compiler_duration = time.time() - start_time
        compiler_runtime.append(compiler_duration)

    n_ranges = [*range(n_low, n_high, n_step)]
    plt.figure()
    plt.plot(n_ranges, compiler_runtime, "ro")
    plt.show()

    plt.figure()
    plt.plot(n_ranges, solver_runtime, "bo")
    plt.show()


if __name__ == "__main__":
    deterministic_solver_runtime(10, 40, 5)
