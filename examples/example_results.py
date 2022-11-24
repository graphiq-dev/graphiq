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

    n_ranges = [*range(2 * n_low, 2 * n_high, 2 * n_step)]
    plt.figure()
    plt.plot(n_ranges, compiler_runtime, "ro")
    plt.show()

    plt.figure()
    plt.plot(n_ranges, solver_runtime, "bo")
    plt.show()


def linear3_example():
    graph = nx.Graph([(1, 2), (2, 3)])
    noise_model = noise_model_loss_and_depolarizing(0.01, 0.01)
    random_numbers = [
        0,
        1,
        5,
        10,
        20,
        30,
        40,
        50,
        60,
        66,
        99,
        100,
        1000,
        2000,
        3000,
        4000,
    ]
    exemplary_multiple_test(graph, noise_model, random_numbers, solver_setting=None)


def linear3_example2():
    graph = nx.Graph([(1, 2), (2, 3)])
    # random_numbers = [*range(20, 100, 5)] + [200, 500]
    random_numbers = [90]
    exemplary_multiple_test(graph, None, random_numbers, solver_setting=None)


def linear4_example():
    graph = nx.Graph([(1, 2), (2, 3), (3, 4)])
    noise_model = noise_model_loss_and_depolarizing(0.05, 0.05)
    random_numbers = [0, 1, 5, 10]
    exemplary_multiple_test(graph, noise_model, random_numbers, solver_setting=None)


def linear5_example():
    graph = nx.Graph([(1, 2), (2, 3), (3, 4), (4, 5)])
    noise_model = noise_model_loss_and_depolarizing(0.1, 0.1)
    random_numbers = [0, 1, 5, 10]
    exemplary_multiple_test(graph, noise_model, random_numbers, solver_setting=None)


def repeater_graph_state_example1():
    graph = repeater_graph_states(3)
    random_numbers = [*range(10, 105, 5)] + [1000, 2000, 3000]
    solver_setting = EvolutionarySearchSolverSetting()
    solver_setting.selection_active = True
    solver_setting.use_adapt_probability = True
    solver_setting.tournament_k = 5
    exemplary_multiple_test(graph, None, random_numbers, solver_setting)


def repeater_graph_state_example2():
    graph = repeater_graph_states(4)
    random_numbers = [*range(10, 105, 5)] + [1000, 2000, 3000]
    solver_setting = EvolutionarySearchSolverSetting()
    solver_setting.selection_active = True
    solver_setting.use_adapt_probability = True
    solver_setting.tournament_k = 10
    solver_setting.n_stop = 60
    exemplary_multiple_test(graph, None, random_numbers, solver_setting)


if __name__ == "__main__":
    # deterministic_solver_runtime(10, 100, 5)
    # linear3_example2()
    # repeater_graph_state_example1()
    repeater_graph_state_example2()
