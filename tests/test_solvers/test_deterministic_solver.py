import pytest
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from src.backends.stabilizer.compiler import StabilizerCompiler
from src.backends.stabilizer.functions.rep_conversion import (
    get_stabilizer_tableau_from_graph,
    get_clifford_tableau_from_graph,
)
from src.solvers.deterministic_solver import DeterministicSolver
from benchmarks.circuits import *
from src.metrics import Infidelity
from src.state import QuantumState


def test_linear4():
    compiler = StabilizerCompiler()
    target_circuit, target = linear_cluster_4qubit_circuit()
    metric = Infidelity(target)
    solver = DeterministicSolver(
        target=target,
        metric=metric,
        compiler=compiler,
    )
    solver.solve()
    score, circuit = solver.hof
    assert np.allclose(score, 0.0)
    circuit.draw_circuit()


def test_square4():
    graph = nx.Graph([(1, 2), (2, 3), (2, 4), (4, 3), (1, 3)])
    target_tableau = get_clifford_tableau_from_graph(graph)
    n_photon = target_tableau.n_qubits
    target = QuantumState(n_photon, target_tableau, representation="stabilizer")
    compiler = StabilizerCompiler()
    metric = Infidelity(target)
    solver = DeterministicSolver(
        target=target,
        metric=metric,
        compiler=compiler,
    )
    solver.solve()
    score, circuit = solver.hof
    assert np.allclose(score, 0.0)
    circuit.draw_circuit()


def test_square4_alternate():
    graph = nx.Graph([(1, 2), (1, 3), (1, 4), (2, 3), (2, 4)])
    # note that adjacency matrix is in the ordering of node creation
    target_tableau = get_clifford_tableau_from_graph(graph)
    n_photon = target_tableau.n_qubits
    target = QuantumState(n_photon, target_tableau, representation="stabilizer")
    compiler = StabilizerCompiler()
    metric = Infidelity(target)
    solver = DeterministicSolver(
        target=target,
        metric=metric,
        compiler=compiler,
    )
    solver.solve()
    score, circuit = solver.hof
    assert np.allclose(score, 0.0)
    circuit.draw_circuit()


def repeater_graph_states(n_inner_qubits):
    edges = []
    for i in range(n_inner_qubits):
        edges.append((2 * i, 2 * i + 1))

    for i in range(n_inner_qubits):
        for j in range(i + 1, n_inner_qubits):
            edges.append((2 * i + 1, 2 * j + 1))
    return nx.Graph(edges)


def test_graph():
    fig, ax = plt.subplots()
    graph = repeater_graph_states(4)
    nx.draw(graph, ax=ax)
    plt.show()


def test_repeater_graph_states():
    graph = repeater_graph_states(4)
    target_tableau = get_clifford_tableau_from_graph(graph)
    n_emitter = DeterministicSolver.determine_n_emitters(target_tableau.to_stabilizer())
    print(f"n_emitter= {n_emitter}")
    n_photon = target_tableau.n_qubits
    target = QuantumState(n_photon, target_tableau, representation="stabilizer")
    compiler = StabilizerCompiler()
    compiler.measurement_determinism = 1
    metric = Infidelity(target)
    solver = DeterministicSolver(
        target=target,
        metric=metric,
        compiler=compiler,
    )
    solver.solve()
    score, circuit = solver.hof
    print(score)
    # assert np.allclose(score, 0.0)
    circuit.draw_circuit()


@pytest.mark.parametrize("n_inner_photons", [4, 5, 6])
def test_graph_states(n_inner_photons):
    target_tableau = get_clifford_tableau_from_graph(
        repeater_graph_states(n_inner_photons)
    )
    n_emitter = DeterministicSolver.determine_n_emitters(target_tableau.to_stabilizer())
    n_photon = target_tableau.n_qubits
    target = QuantumState(n_photon, target_tableau, representation="stabilizer")
    compiler = StabilizerCompiler()
    metric = Infidelity(target)
    solver = DeterministicSolver(
        target=target,
        metric=metric,
        compiler=compiler,
    )
    solver.solve()
    score, circuit = solver.hof
    print(score)
    # assert np.allclose(score, 0.0)
    # circuit.draw_circuit()
