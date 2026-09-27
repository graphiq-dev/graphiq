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
import pytest

import graphiq.backends.density_matrix.functions as dmf
import graphiq.circuit.ops as ops
from graphiq.backends.graph.state import Graph, MixedGraph


def test_graph_initialization():
    graph = nx.Graph([(0, 1), (1, 2), (1, 3), (2, 3)])
    state = Graph(graph)
    state.draw()
    assert state.n_qubits == 4
    print(f"Neighbors of node 1 are {state.get_neighbors(1)} ")
    lc_state = state.local_complementation(1, copy=True)
    state.draw()
    lc_state.draw()


def test_graph_lc():
    lc_dict = {0: ["x"], 1: ["h"], 2: ["s"], 3: ["s", "h"]}
    graph = nx.Graph([(0, 1), (1, 2), (1, 3), (2, 3)])
    state = Graph(graph, lc_dict)
    assert state.find_lc(1) == [ops.Hadamard, ops.Identity]
    assert dmf.check_equivalent_unitaries(
        ops.local_clifford_to_matrix_map(state.find_lc(3)), dmf.phase() @ dmf.hadamard()
    )


def test_mixed_graph():
    graph1 = nx.Graph([(0, 1), (1, 2), (1, 3), (2, 3)])
    state1 = Graph(graph1)
    graph2 = nx.Graph([(0, 2), (1, 2), (0, 3), (2, 3)])
    state2 = Graph(graph2)
    state3 = MixedGraph([(0.5, state1), (0.5, state2)])
    print(state3.mixture)
    assert state3.n_qubits == 4
    assert state3.probability == 1.0


def test_find_lc_missing_node_raises():
    state = Graph(nx.Graph([(0, 1)]))
    with pytest.raises(ValueError):
        state.find_lc(5)


def test_update_lc_missing_node_raises():
    state = Graph(nx.Graph([(0, 1)]))
    with pytest.raises(ValueError):
        state.update_lc(5, ["h"])


def test_update_lc_sets_gate_and_resets_to_identity():
    state = Graph(nx.Graph([(0, 1)]))
    assert state.find_lc(0) == [ops.Identity]
    state.update_lc(0, ["h"])
    assert state.find_lc(0) != [ops.Identity]
    state.update_lc(0, None)
    assert state.find_lc(0) == [ops.Identity]


def test_add_node_new_node_and_existing_node_is_noop():
    state = Graph(nx.Graph([(0, 1)]))
    state.add_node(2)
    assert 2 in state.get_nodes()
    assert state.find_lc(2) == [ops.Identity]

    # a node already present is left untouched, including its local Clifford
    state.update_lc(2, ["h"])
    state.add_node(2)
    assert state.find_lc(2) != [ops.Identity]


def test_add_node_with_explicit_lc_gate_stores_it():
    state = Graph(nx.Graph([(0, 1)]))
    state.add_node(3, ["h"])
    assert 3 in state.get_nodes()
    assert state.find_lc(3) == ops.simplify_local_clifford([ops.name_to_class_map("h")])


def test_add_edge_creates_missing_second_endpoint():
    state = Graph(nx.Graph([(0, 1)]))
    state.add_edge(1, 5)
    assert 5 in state.get_nodes()
    assert {1, 5} in [set(e) for e in state.get_edges()]


def test_add_edge_creates_missing_first_endpoint():
    state = Graph(nx.Graph([(0, 1)]))
    state.add_edge(5, 1)
    assert 5 in state.get_nodes()
    assert state.find_lc(5) == [ops.Identity]
    assert {5, 1} in [set(e) for e in state.get_edges()]


def test_get_edges_and_get_nodes():
    state = Graph(nx.Graph([(0, 1), (1, 2)]))
    assert set(state.get_nodes()) == {0, 1, 2}
    assert {frozenset(e) for e in state.get_edges()} == {
        frozenset((0, 1)),
        frozenset((1, 2)),
    }


def test_get_neighbors_missing_node_raises():
    state = Graph(nx.Graph([(0, 1)]))
    with pytest.raises(ValueError):
        state.get_neighbors(99)


def test_mixed_graph_constructor_rejects_bare_nx_graph():
    with pytest.raises(TypeError):
        MixedGraph(nx.Graph([(0, 1)]))


def test_mixed_graph_mixture_setter_rejects_bare_nx_graph():
    graph1 = Graph(nx.Graph([(0, 1)]))
    state = MixedGraph(graph1)
    with pytest.raises(TypeError):
        state.mixture = nx.Graph([(0, 1)])


def test_mixed_graph_rejects_list_of_bare_nx_graph():
    with pytest.raises(AssertionError):
        MixedGraph([(1.0, nx.Graph([(0, 1)]))])


def test_mixed_graph_mixture_setter_list():
    graph1 = Graph(nx.Graph([(0, 1)]))
    graph2 = Graph(nx.Graph([(0, 1)]))
    state = MixedGraph(graph1)
    state.mixture = [(0.5, graph1), (0.5, graph2)]
    assert state.mixture == [(0.5, graph1), (0.5, graph2)]


def test_mixed_graph_mixture_setter_single_graph():
    graph1 = Graph(nx.Graph([(0, 1)]))
    graph2 = Graph(nx.Graph([(0, 1), (1, 2)]))
    state = MixedGraph(graph1)
    state.mixture = graph2
    assert state.mixture == [(1.0, graph2)]


def test_mixed_graph_data_setter_delegates_to_mixture():
    graph1 = Graph(nx.Graph([(0, 1)]))
    graph2 = Graph(nx.Graph([(0, 1)]))
    state = MixedGraph(graph1)
    state.data = [(0.5, graph1), (0.5, graph2)]
    assert state.data == [(0.5, graph1), (0.5, graph2)]
