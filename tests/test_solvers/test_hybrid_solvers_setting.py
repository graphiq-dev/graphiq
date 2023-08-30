import pytest
import numpy as np
import networkx as nx
from benchmarks.graph_states import *

from src.backends.stabilizer.compiler import StabilizerCompiler
from src.backends.stabilizer.functions.rep_conversion import (
    get_clifford_tableau_from_graph,
)
from src.solvers.deterministic_solver import DeterministicSolver
from src.solvers.evolutionary_solver import EvolutionarySearchSolverSetting
from src.solvers.hybrid_solvers import (
    HybridEvolutionarySolver,
    HybridGraphSearchSolver,
    HybridGraphSearchSolverSetting,
)
from benchmarks.circuits import *
from src.metrics import Infidelity
from src.state import QuantumState
from benchmarks.alternate_circuits import *
import src.noise.noise_models as noise
import time


# Test HybridGraphSearchSolverSetting class
def test_hybrid_graph_search_solver_setting_init():
    solver_settings = HybridGraphSearchSolverSetting()

    assert solver_settings.n_lc_graphs == 10
    assert solver_settings.n_iso_graphs == 10
    assert not solver_settings.base_solver_setting
    assert not solver_settings.verbose
    assert solver_settings.save_openqasm == "none"


# Test solver setting related function
# Test for HybridGraphSearchSolver.get_iso_graph_from_setting()
def test_hybrid_graph_search_get_iso_graphs():
    target_graph = nx.star_graph(3)

    target_tableau = get_clifford_tableau_from_graph(target_graph)
    n_photon = target_tableau.n_qubits
    target_state = QuantumState(target_tableau, rep_type="s")

    compiler = StabilizerCompiler()

    metric = Infidelity(target=target_state)
    solver_setting = HybridGraphSearchSolverSetting(n_iso_graphs=2)

    solver = HybridGraphSearchSolver(
        target=target_state,
        metric=metric,
        compiler=compiler,
        graph_solver_setting=solver_setting,
        base_solver=DeterministicSolver,
    )

    adj_matrix = nx.to_numpy_array(target_graph)

    assert len(solver.get_iso_graph_from_setting(adj_matrix)) == 2


# Test for HybridGraphSearchSolver.get_lc_graph_from_setting()
def test_hybrid_graph_search_get_lc_graphs():
    target_graph = nx.star_graph(3)

    target_tableau = get_clifford_tableau_from_graph(target_graph)
    n_photon = target_tableau.n_qubits
    target_state = QuantumState(target_tableau, rep_type="s")

    compiler = StabilizerCompiler()

    metric = Infidelity(target=target_state)
    solver_setting = HybridGraphSearchSolverSetting(n_lc_graphs=2)

    solver = HybridGraphSearchSolver(
        target=target_state,
        metric=metric,
        compiler=compiler,
        graph_solver_setting=solver_setting,
        base_solver=DeterministicSolver,
    )

    adj_matrix = nx.to_numpy_array(target_graph)

    assert len(solver.get_lc_graph_from_setting(adj_matrix)) == 2


def test_hybird_graph_search_solver_callback_func():
    # Test solver callback function to add attributes to solver result
    def circuit_depth(circuit, target_state):
        return circuit.depth

    def n_emitters(circuit, target_state):
        return circuit.n_emitters

    def emitter_depth(circuit, target_state):
        # Use case to add multiple columns to solver result including timing
        depth = circuit.calculate_reg_depth("e")
        t0 = time.time()

        d = {
            "max_emitter_depth": max(depth),
            "min_emitter_depth": min(depth),
            "emitter_depth_run_time": time.time() - t0,
        }
        return d

    test_mapping = {
        "circuit_depth": circuit_depth,
        "n_emitters": n_emitters,
        "emitter_depth": emitter_depth,
    }

    target_graph = linear_cluster_state(4)
    target_graph = target_graph.data

    target_tableau = get_clifford_tableau_from_graph(target_graph)
    n_photon = target_tableau.n_qubits
    target_state = QuantumState(target_tableau, rep_type="s")

    compiler = StabilizerCompiler()

    metric = Infidelity(target=target_state)
    solver_setting = HybridGraphSearchSolverSetting(
        n_iso_graphs=5, n_lc_graphs=5, callback_func=test_mapping
    )

    solver = HybridGraphSearchSolver(
        target=target_state,
        metric=metric,
        compiler=compiler,
        graph_solver_setting=solver_setting,
        base_solver=DeterministicSolver,
    )

    circuit_data = solver.solve()
    assert circuit_data.columns == [
        "circuit",
        "target_state",
        "score",
        "circuit_depth",
        "n_emitters",
        "max_emitter_depth",
        "min_emitter_depth",
        "emitter_depth_run_time",
    ]
