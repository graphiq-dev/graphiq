import pytest
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from benchmarks.graph_states import repeater_graph_states
from src.backends.stabilizer.compiler import StabilizerCompiler
from src.backends.stabilizer.functions.rep_conversion import (
    get_clifford_tableau_from_graph,
)
from src.solvers.hybrid_solvers import HybridEvolutionarySolver
from benchmarks.circuits import *
from src.metrics import Infidelity
from src.state import QuantumState


def graph_stabilizer_setup(graph, solver_class, random_seed):
    """
    A common piece of code to run a graph state with the stabilizer backend

    :param graph: the graph representation of a graph state
    :type graph: networkX.Graph
    :param solver_class: a solver class to run
    :type solver_class: a subclass of SolverBase
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
    )
    solver.seed(random_seed)
    solver.solve()
    return solver


def test_repeater_graph_state_4():
    graph = repeater_graph_states(4)
    solver = graph_stabilizer_setup(graph, HybridEvolutionarySolver, 1)
    score, circuit = solver.result
    assert np.allclose(score, 0.0)
    # circuit.draw_circuit()


def test_square4():
    graph = nx.Graph([(1, 2), (2, 3), (2, 4), (4, 3), (1, 3)])
    solver = graph_stabilizer_setup(graph, HybridEvolutionarySolver, 1)
    score, circuit = solver.result
    assert np.allclose(score, 0.0)
    # circuit.draw_circuit()
