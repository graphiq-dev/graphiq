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

from graphiq.backends.graph.state import Graph
from graphiq.backends.lc_equivalence_check import is_lc_equivalent
from graphiq.benchmarks import graph_states as gs


def _build_reference_bi_repeater_graph(n):
    """
    Hand-built reference for the biclique repeater graph state: a complete
    bipartite graph (biclique) between n "left" inner qubits and n "right"
    inner qubits, with exactly one pendant leaf qubit attached to each inner
    qubit. Node labels are independent of the ones used by
    gs.bi_repeater_graph_states; only the graph structure is compared.
    """
    left = [f"left_{i}" for i in range(n)]
    right = [f"right_{i}" for i in range(n)]
    left_leaves = [f"left_leaf_{i}" for i in range(n)]
    right_leaves = [f"right_leaf_{i}" for i in range(n)]
    graph = nx.Graph()
    graph.add_edges_from((left[i], left_leaves[i]) for i in range(n))
    graph.add_edges_from((right[i], right_leaves[i]) for i in range(n))
    graph.add_edges_from((left[i], right[j]) for i in range(n) for j in range(n))
    return graph


def test_bi_repeater_graph_states():
    n = 3
    graph = gs.bi_repeater_graph_states(n)
    reference = _build_reference_bi_repeater_graph(n)
    assert nx.is_connected(graph)
    assert nx.is_isomorphic(graph, reference)


@pytest.mark.parametrize("n_list", [[2, 3], [2, 3, 2], [3, 3, 3], [1, 2, 1]])
def test_crazy(n_list):
    graph = gs.crazy(n_list)
    assert graph.number_of_nodes() == sum(n_list)

    # consecutive layers are complete bipartite and there are no intra-layer edges
    layers = []
    start = 0
    for n in n_list:
        layers.append(list(range(start, start + n)))
        start += n
    for layer in layers:
        for u in layer:
            for v in layer:
                if u != v:
                    assert not graph.has_edge(u, v)
    for layer_a, layer_b in zip(layers, layers[1:]):
        for u in layer_a:
            for v in layer_b:
                assert graph.has_edge(u, v)

    alternate_graph = gs.crazy(n_list, alternate_order=True)
    assert nx.is_isomorphic(graph, alternate_graph)


@pytest.mark.parametrize("row_col", [(2, 2), (2, 3), (3, 3), (3, 4)])
def test_two_d_cluster(row_col):
    graph = gs.two_d_cluster(row_col)
    reference = nx.grid_2d_graph(*row_col)
    assert nx.is_isomorphic(graph, reference)


@pytest.mark.parametrize("row_col_depth", [(2, 2, 2), (2, 3, 2), (3, 3, 3), (2, 3, 4)])
def test_three_d_cluster(row_col_depth):
    graph = gs.three_d_cluster(row_col_depth)
    row, col, depth = row_col_depth
    reference = nx.grid_graph([depth, col, row])
    assert nx.is_isomorphic(graph, reference)


@pytest.mark.parametrize("b_vector", [[2], [2, 3], [3, 2, 1]])
def test_branching_tree(b_vector):
    graph = gs.branching_tree(b_vector)
    expected_nodes = 1
    running_product = 1
    for b in b_vector:
        running_product *= b
        expected_nodes += running_product
    assert graph.number_of_nodes() == expected_nodes
    assert nx.is_tree(graph)


def test_random_graph_state():
    state = gs.random_graph_state(6, 0.5, np_rng=np.random.default_rng(seed=42))
    assert isinstance(state, Graph)
    graph = state.data
    assert graph.number_of_nodes() == 6
    assert nx.is_connected(graph)

    state_again = gs.random_graph_state(6, 0.5, np_rng=np.random.default_rng(seed=42))
    assert nx.utils.graphs_equal(graph, state_again.data)


def test_random_lc_equivalent():
    start_graph = Graph(nx.path_graph(5))
    new_graphs, _ = gs.random_lc_equivalent(
        start_graph, 4, 3, np_rng=np.random.default_rng(seed=7)
    )
    start_adj = nx.to_numpy_array(start_graph.data).astype(int)
    for new_graph in new_graphs:
        new_adj = nx.to_numpy_array(new_graph.data).astype(int)
        equivalent, _ = is_lc_equivalent(start_adj, new_adj, mode="deterministic")
        assert equivalent
