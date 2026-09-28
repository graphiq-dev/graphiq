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

from graphiq.utils.preprocessing import _select_graphs


def _make_graph(label):
    graph = nx.Graph()
    graph.add_node(label)
    return graph


def test_select_graphs_keeps_the_true_lowest_entries():
    # With limit=2, feeding metric values 5.0, then 3.0, then 4.0 must keep the two lowest
    # (3.0 and 4.0), not evict 3.0 -- the best score seen -- once the list is at capacity.
    graph_5 = _make_graph("five")
    graph_3 = _make_graph("three")
    graph_4 = _make_graph("four")

    candidate_graphs = []
    candidate_graphs = _select_graphs(candidate_graphs, graph_5, 2, 5.0)
    candidate_graphs = _select_graphs(candidate_graphs, graph_3, 2, 3.0)
    candidate_graphs = _select_graphs(candidate_graphs, graph_4, 2, 4.0)

    assert candidate_graphs == [(3.0, graph_3), (4.0, graph_4)]


def test_select_graphs_tolerates_tied_metric_values():
    # Equal metric values are common (integer-valued edge/eccentricity counts); comparing the
    # graphs themselves as a tiebreak must never happen, since nx.Graph has no ordering.
    graph_a = _make_graph("a")
    graph_b = _make_graph("b")
    graph_c = _make_graph("c")

    candidate_graphs = []
    candidate_graphs = _select_graphs(candidate_graphs, graph_a, 2, 3.0)
    candidate_graphs = _select_graphs(candidate_graphs, graph_b, 2, 3.0)
    candidate_graphs = _select_graphs(candidate_graphs, graph_c, 2, 3.0)

    assert len(candidate_graphs) == 2
    assert all(value == 3.0 for value, _ in candidate_graphs)


def test_select_graphs_keeps_starting_graph_when_it_is_best():
    # _select_graphs is seeded with the starting graph; when its metric is the lowest seen,
    # it must still be present in the result.
    starting_graph = _make_graph("start")
    other_graph_1 = _make_graph("other1")
    other_graph_2 = _make_graph("other2")

    candidate_graphs = [(1.0, starting_graph)]
    candidate_graphs = _select_graphs(candidate_graphs, other_graph_1, 2, 5.0)
    candidate_graphs = _select_graphs(candidate_graphs, other_graph_2, 2, 4.0)

    assert candidate_graphs[0] == (1.0, starting_graph)
    assert starting_graph in [graph for _, graph in candidate_graphs]
