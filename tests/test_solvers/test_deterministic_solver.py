import pytest

from benchmarks.graph_states import repeater_graph_states
from src.backends.stabilizer.compiler import StabilizerCompiler
from src.backends.density_matrix.compiler import DensityMatrixCompiler
from src.backends.density_matrix.functions import fidelity
from src.backends.state_rep_conversion import graph_to_density
from src.backends.stabilizer.functions.rep_conversion import (
    get_clifford_tableau_from_graph,
)
from src.solvers.deterministic_solver import DeterministicSolver
from benchmarks.circuits import *
from src.metrics import Infidelity
from src.state import QuantumState
import networkx as nx
import numpy as np


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
    score, circuit = solver.result
    assert np.allclose(score, 0.0)
    circuit.draw_circuit()


def test_square4():
    graph = nx.Graph([(1, 2), (2, 3), (2, 4), (4, 3), (1, 3)])
    target_tableau = get_clifford_tableau_from_graph(graph)
    n_photon = target_tableau.n_qubits
    target = QuantumState(target_tableau, rep_type="stab")
    compiler = StabilizerCompiler()
    compiler.measurement_determinism = 1
    metric = Infidelity(target)
    solver = DeterministicSolver(
        target=target,
        metric=metric,
        compiler=compiler,
    )
    solver.solve()
    score, circuit = solver.result
    assert np.allclose(score, 0.0)
    circuit.draw_circuit()


def test_square4_alternate():
    graph = nx.Graph([(1, 2), (1, 3), (1, 4), (2, 3), (2, 4)])
    # note that adjacency matrix is in the ordering of node creation
    target_tableau = get_clifford_tableau_from_graph(graph)
    n_photon = target_tableau.n_qubits
    target = QuantumState(target_tableau, rep_type="stab")
    compiler = StabilizerCompiler()
    metric = Infidelity(target)
    solver = DeterministicSolver(
        target=target,
        metric=metric,
        compiler=compiler,
    )
    solver.solve()
    score, circuit = solver.result
    circuit.draw_circuit()
    assert np.allclose(score, 0.0)


def test_repeater_graph_state_4():
    graph = repeater_graph_states(4)
    target_tableau = get_clifford_tableau_from_graph(graph)
    n_photon = target_tableau.n_qubits
    target = QuantumState(target_tableau, rep_type="stab")
    compiler = StabilizerCompiler()
    compiler.measurement_determinism = 1
    metric = Infidelity(target)
    solver = DeterministicSolver(
        target=target,
        metric=metric,
        compiler=compiler,
    )
    solver.solve()
    score, circuit = solver.result
    assert np.allclose(score, 0.0)
    # circuit.draw_circuit()


@pytest.mark.parametrize("n_inner_photons", [6, 7, 8, 9, 10])
def test_repeater_graph_states(n_inner_photons):
    target_tableau = get_clifford_tableau_from_graph(
        repeater_graph_states(n_inner_photons)
    )
    n_photon = target_tableau.n_qubits
    target = QuantumState(target_tableau, rep_type="stab")
    compiler = StabilizerCompiler()
    metric = Infidelity(target)
    solver = DeterministicSolver(
        target=target,
        metric=metric,
        compiler=compiler,
    )
    solver.solve()
    score, circuit = solver.result
    assert np.allclose(score, 0.0)
    # circuit.draw_circuit()


@pytest.mark.parametrize("n_nodes", [3, 4, 5, 6])
def test_random_graph_states(n_nodes):
    not_connected = True
    while not_connected:
        graph = nx.fast_gnp_random_graph(n_nodes, 0.5)
        not_connected = not nx.is_connected(graph)

    target_tableau = get_clifford_tableau_from_graph(graph)
    n_photon = target_tableau.n_qubits
    target = QuantumState(target_tableau, rep_type="stab")
    compiler = StabilizerCompiler()
    metric = Infidelity(target)
    solver = DeterministicSolver(
        target=target,
        metric=metric,
        compiler=compiler,
    )
    solver.solve()
    score, circuit = solver.result
    assert np.allclose(score, 0.0)

    compiler = DensityMatrixCompiler()
    # compiler.measurement_determinism = 1
    target_state = graph_to_density(graph)
    final_state = compiler.compile(circuit)
    n_total = final_state.n_qubits
    # keeping only photonic qubits in the circuit output
    final_state.partial_trace([*range(n_nodes)], n_total * [2])
    fid = fidelity(target_state, np.round(final_state.rep_data.data, 8))
    assert fid > 0.99
