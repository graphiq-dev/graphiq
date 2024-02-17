# Copyright (c) 2022-2024 Quantum Bridge Technologies Inc.
# Copyright (c) 2022-2024 Ki3 Photonics Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import networkx as nx
import numpy as np
import pytest

from graphiq.benchmarks.graph_states import repeater_graph_states
from graphiq.backends.density_matrix.compiler import DensityMatrixCompiler
from graphiq.backends.density_matrix.functions import fidelity
from graphiq.backends.stabilizer.compiler import StabilizerCompiler
from graphiq.backends.stabilizer.functions.rep_conversion import (
    get_clifford_tableau_from_graph,
)
from graphiq.backends.state_rep_conversion import graph_to_density
from graphiq.metrics import Infidelity
from graphiq.solvers.time_reversed_solver import TimeReversedSolver
from graphiq.state import QuantumState
from graphiq.benchmarks.circuits import linear_cluster_4qubit_circuit


def test_linear4():
    compiler = StabilizerCompiler()
    target_circuit, target = linear_cluster_4qubit_circuit()
    metric = Infidelity(target)
    solver = TimeReversedSolver(
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
    solver = TimeReversedSolver(
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
    solver = TimeReversedSolver(
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
    solver = TimeReversedSolver(
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
    solver = TimeReversedSolver(
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
    solver = TimeReversedSolver(
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
