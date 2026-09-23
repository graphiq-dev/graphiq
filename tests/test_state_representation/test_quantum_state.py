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

from graphiq.backends.density_matrix import numpy as dmnp
import graphiq.backends.state_rep_conversion as rc
from graphiq.backends.graph.state import Graph, MixedGraph
from graphiq.backends.stabilizer.clifford_tableau import CliffordTableau
from graphiq.backends.stabilizer.functions.clifford import create_n_ket0_state
from graphiq.backends.stabilizer.functions.transformation import (
    cnot_gate,
    hadamard_gate,
)
from graphiq.state import QuantumState


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
def test_initializing_dm_1(n):
    data = dmnp.eye(2**n)
    state = QuantumState(data, rep_type="dm")
    assert np.allclose(state.rep_data.data, data / np.trace(data))


def test_initializing_dm_2():
    """Test that it fails if we don't provide the right datatype"""
    data = "bad input"
    with pytest.raises(TypeError):
        QuantumState(data, rep_type="dm")


def test_initializing_graph_2():
    n = 3
    data = nx.Graph([(1, 2), (2, 3)])
    state = QuantumState(data, rep_type="g")
    assert len(state.rep_data.data.nodes) == n


def test_initializing_stabilizer_1():
    """Checks that we can initialize stabilizers with integers"""
    n = 130
    state = QuantumState(n, rep_type="s")
    assert state.rep_data.n_qubits == n


def test_initializing_stabilizer_2():
    """Checks that we can initialize stabilizers with a CliffordTableau"""
    n = 130
    state = QuantumState(CliffordTableau(n), rep_type="s")
    assert state.rep_data.n_qubits == n


def test_convert_representation_graph_stabilizer_graph_round_trip_pure_arm():
    graph = nx.path_graph(3)
    state = QuantumState(graph, rep_type="g")

    state.convert_representation("s")
    state.convert_representation("g")

    assert nx.is_isomorphic(state.rep_data.data, graph)


def test_convert_representation_stabilizer_to_graph_mixed_arm():
    path3 = nx.path_graph(3)
    star3 = nx.star_graph(2)
    mixture = [
        (0.7, CliffordTableau(rc.graph_to_stabilizer(path3)[0][1])),
        (0.3, CliffordTableau(rc.graph_to_stabilizer(star3)[0][1])),
    ]
    state = QuantumState(mixture, rep_type="s", mixed=True)

    state.convert_representation("g")

    mixed_graph = state.rep_data
    assert isinstance(mixed_graph, MixedGraph)
    assert [p_i for p_i, _ in mixed_graph.mixture] == [0.7, 0.3]
    assert nx.is_isomorphic(mixed_graph.mixture[0][1].data, path3)
    assert nx.is_isomorphic(mixed_graph.mixture[1][1].data, star3)


def test_str_of_a_mixed_graph():
    # MixedGraph never sets _data, so the base class __str__ must go through data
    mixed_graph = MixedGraph([(1.0, Graph(nx.path_graph(2)))])
    assert str(mixed_graph).startswith("MixedGraph")


def test_graph_to_density_mixed_arm_reached_through_the_private_helper():
    # QuantumState.validate_data has no MixedGraph arm, so a MixedGraph cannot enter a
    # QuantumState through the constructor at all -- the mixed-graph conversion path can only be
    # reached by calling the private helper directly on an instance whose "mixed" flag is set.
    qs = QuantumState(nx.path_graph(3), rep_type="g", mixed=True)
    graph_a = nx.path_graph(3)
    graph_b = nx.cycle_graph(3)
    mixed_graph = MixedGraph([(0.7, Graph(graph_a)), (0.3, Graph(graph_b))])

    density = qs._graph_to_density(mixed_graph)

    expected = 0.7 * rc.graph_to_density(graph_a) + 0.3 * rc.graph_to_density(graph_b)
    assert np.allclose(density.data, expected)


def test_convert_representation_graph_to_density_matrix_pure_arm():
    graph = nx.path_graph(3)
    state = QuantumState(graph, rep_type="g")

    state.convert_representation("dm")

    assert np.allclose(state.rep_data.data, rc.graph_to_density(graph))


def test_convert_representation_graph_to_stabilizer_pure_arm():
    graph = nx.path_graph(3)
    state = QuantumState(graph, rep_type="g")

    state.convert_representation("s")

    expected = rc.graph_to_stabilizer(graph)[0][1]
    assert state.rep_data.data.to_stabilizer() == expected


@pytest.mark.parametrize("outcome", [0, 1])
def test_partial_trace_forwards_measurement_determinism_to_stabilizer(outcome):
    # Tracing three qubits out of GHZ4 leaves |outcome> on the survivor, so the
    # result shows whether the forced outcome reached the backend.
    tableau = hadamard_gate(create_n_ket0_state(4), 0)
    for target in (1, 2, 3):
        tableau = cnot_gate(tableau, 0, target)
    state = QuantumState(tableau, rep_type="s")

    state.partial_trace(keep=[0], dims=4 * [2], measurement_determinism=outcome)

    rho = rc.stabilizer_to_density(state.rep_data.tableau.to_stabilizer())
    assert np.allclose(rho, np.diag([1.0 - outcome, float(outcome)]))


def test_density_matrix_partial_trace_accepts_measurement_determinism():
    bell = np.array([1, 0, 0, 1]) / np.sqrt(2)
    state = QuantumState(np.outer(bell, bell), rep_type="dm")

    state.partial_trace(keep=[0], dims=[2, 2], measurement_determinism=1)

    assert np.allclose(state.rep_data.data, np.eye(2) / 2)
