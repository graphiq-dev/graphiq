import pytest
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from benchmarks.graph_states import repeater_graph_states
from src.backends.stabilizer.compiler import StabilizerCompiler
from src.backends.stabilizer.functions.rep_conversion import (
    get_clifford_tableau_from_graph,
)
from src.solvers.evolutionary_solver import EvolutionarySolverSetting
from src.solvers.hybrid_solvers import HybridEvolutionarySolver
from benchmarks.circuits import *
from src.metrics import Infidelity
from src.state import QuantumState
from benchmarks.alternate_circuits import *


def graph_stabilizer_setup(graph, solver_class, solver_setting, random_seed):
    """
    A common piece of code to run a graph state with the stabilizer backend

    :param graph: the graph representation of a graph state
    :type graph: networkX.Graph
    :param solver_class: a solver class to run
    :type solver_class: a subclass of SolverBase
    :param solver_setting:
    :type solver_setting:
    :param random_seed: a random seed
    :type random_seed: int
    :return: the solver instance
    :rtype: SolverBase
    """
    target_tableau = get_clifford_tableau_from_graph(graph)
    n_photon = target_tableau.n_qubits
    target = QuantumState(n_photon, target_tableau, representation="stabilizer")
    compiler = StabilizerCompiler()
    compiler.measurement_determinism = 1
    metric = Infidelity(target)
    solver = solver_class(
        target=target,
        metric=metric,
        compiler=compiler,
        solver_setting=solver_setting,
    )
    solver.seed(random_seed)
    solver.solve()
    return solver


def test_repeater_graph_state_4():
    graph = repeater_graph_states(4)
    solver_setting = EvolutionarySolverSetting()
    solver = graph_stabilizer_setup(
        graph, HybridEvolutionarySolver, solver_setting, 1000
    )
    score, circuit = solver.result
    assert np.allclose(score, 0.0)
    # circuit.draw_circuit()


def test_square4():
    graph = nx.Graph([(1, 2), (2, 3), (2, 4), (4, 3), (1, 3)])
    solver_setting = EvolutionarySolverSetting()
    solver = graph_stabilizer_setup(graph, HybridEvolutionarySolver, solver_setting, 2)
    score, circuit = solver.result
    assert np.allclose(score, 0.0)
    # circuit.draw_circuit()


def test_alternate_circuits_1():
    results = run_one_repeater_graph_state(3, 0)
    if len(results) > 1:
        print(f"Find {len(results)} circuits that produce the same state.")
        for i in range(len(results)):
            results[i].draw_circuit()


def test_alternate_circuits_2():
    results = run_one_repeater_graph_state(4, 1)
    if len(results) > 1:
        print(f"Find {len(results)} circuits that produce the same state.")
        for i in range(len(results)):
            results[i].draw_circuit()


def test_alternate_circuits_w_noise():
    results = run_one_repeater_graph_state_w_loss(4, 0, 0, 1000)
    if len(results) > 1:
        print(f"Find {len(results)} circuits that produce the same state.")
        for i in range(len(results)):
            results[i].draw_circuit()


def test_alternate_circuits_w_noise2():
    results = run_one_repeater_graph_state_w_loss(
        4, loss_rate=0.01, error_rate=0.01, random_seed=1000
    )
    if len(results) > 1:
        print(f"Find {len(results)} circuits that produce the same state.")
        for i in range(len(results)):
            results[i].draw_circuit()
