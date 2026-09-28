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
from graphiq.backends.stabilizer.tableau import StabilizerTableau
from graphiq.backends.stabilizer.functions.rep_conversion import (
    get_clifford_tableau_from_graph,
)
from graphiq.backends.state_rep_conversion import graph_to_density
from graphiq.metrics import Infidelity
from graphiq.solvers.time_reversed_solver import TimeReversedSolver
from graphiq.state import QuantumState
from graphiq.benchmarks.circuits import linear_cluster_4qubit_circuit
from graphiq.circuit import ops


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


def _one_qubit_tableau(pauli):
    x = np.array([[1 if pauli in "XY" else 0]])
    z = np.array([[1 if pauli in "YZ" else 0]])
    return StabilizerTableau([x, z], np.array([0]))


def _cell(tableau):
    return {(1, 0): "X", (1, 1): "Y", (0, 1): "Z", (0, 0): "I"}[
        (tableau.x_matrix[0, 0], tableau.z_matrix[0, 0])
    ]


@pytest.mark.parametrize(
    "current, result, expected_gates, expected_final",
    [
        ("X", "x", [], "X"),
        ("X", "y", [ops.SigmaZ, ops.Phase], "Y"),
        ("X", "z", [ops.Hadamard], "Z"),
        ("Y", "x", [ops.Phase], "X"),
        ("Y", "y", [], "Y"),
        ("Y", "z", [ops.Phase, ops.Hadamard], "Z"),
        ("Z", "x", [ops.Hadamard], "X"),
        ("Z", "y", [ops.Hadamard, ops.SigmaZ, ops.Phase], "Y"),
        ("Z", "z", [], "Z"),
        ("X", "Z", [ops.Hadamard], "Z"),
        ("Z", "X", [ops.Hadamard], "X"),
    ],
)
def test_change_pauli_type_rotates_between_the_three_bases(
    current, result, expected_gates, expected_final
):
    solver = TimeReversedSolver.__new__(TimeReversedSolver)
    tableau = _one_qubit_tableau(current)

    gates = solver._change_pauli_type(tableau, 0, 0, result)

    assert gates == expected_gates
    assert _cell(tableau) == expected_final


def test_change_pauli_type_leaves_an_identity_entry_alone():
    solver = TimeReversedSolver.__new__(TimeReversedSolver)
    tableau = _one_qubit_tableau("I")

    assert solver._change_pauli_type(tableau, 0, 0, "z") == []
    assert _cell(tableau) == "I"


@pytest.mark.parametrize("pauli", ["X", "I"])
@pytest.mark.parametrize("result", ["q", "", "zz", "i"])
def test_change_pauli_type_rejects_an_unrecognised_basis(result, pauli):
    solver = TimeReversedSolver.__new__(TimeReversedSolver)
    tableau = _one_qubit_tableau(pauli)

    with pytest.raises(ValueError, match=repr(result)):
        solver._change_pauli_type(tableau, 0, 0, result)
