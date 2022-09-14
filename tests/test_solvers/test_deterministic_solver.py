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
    assert np.allclose(score, 0.0)
    circuit.draw_circuit()


def test_square4():
    graph = nx.Graph([(1, 2), (2, 3), (2, 4), (4, 3), (1, 3)])
    target_tableau = get_clifford_tableau_from_graph(graph)
    n_emitter = 2
    n_photon = 4
    target = QuantumState(n_photon, target_tableau, representation="stabilizer")
    compiler = StabilizerCompiler()
    metric = Infidelity(target)
    solver = DeterministicSolver(
        target=target,
        metric=metric,
        compiler=compiler,
        n_emitter=n_emitter,
        n_photon=n_photon,
    )
    solver.solve()
    score, circuit = solver.hof
    assert np.allclose(score, 0.0)
    circuit.draw_circuit()


def test_square4_alternate():
    graph = nx.Graph([(1, 2), (1, 3), (1, 4), (2, 3), (2, 4)])
    # note that adjacency matrix is in the ordering of node creation
    target_tableau = get_clifford_tableau_from_graph(graph)
    n_emitter = 1
    n_photon = 4
    target = QuantumState(n_photon, target_tableau, representation="stabilizer")
    compiler = StabilizerCompiler()
    metric = Infidelity(target)
    solver = DeterministicSolver(
        target=target,
        metric=metric,
        compiler=compiler,
        n_emitter=n_emitter,
        n_photon=n_photon,
    )
    solver.solve()
    score, circuit = solver.hof
    assert np.allclose(score, 0.0)
    circuit.draw_circuit()