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
"""
Regression tests for the graph-picking behaviour of the three heuristic demos in
graphiq.benchmarks.lc_equivalence. Each demo is supposed to draw and search a graph, then
draw and search a second graph reached from it by local complementation; this module guards
that the second graph is actually distinct from the first, rather than the input repeated.
"""

import networkx as nx
import pytest

from graphiq.benchmarks import lc_equivalence
from graphiq.utils.preprocessing import get_lc_graph_by_max_edge, graph_metric_lists


def _spy_on_search_and_draw(monkeypatch):
    calls = []

    def fake_exemplary_run(graph, *args, **kwargs):
        calls.append(graph)
        return None

    monkeypatch.setattr(lc_equivalence, "exemplary_run", fake_exemplary_run)
    monkeypatch.setattr(
        lc_equivalence, "report_alternate_circuits", lambda result: None
    )
    monkeypatch.setattr(nx, "draw", lambda *args, **kwargs: None)
    return calls


def test_heuristic_example1_searches_a_distinct_second_graph(monkeypatch):
    calls = _spy_on_search_and_draw(monkeypatch)

    lc_equivalence.heuristic_example1()

    assert len(calls) == 2
    first_edges = set(map(frozenset, calls[0].edges()))
    second_edges = set(map(frozenset, calls[1].edges()))
    assert first_edges != second_edges


def test_heuristic_example2_searches_a_distinct_second_graph(monkeypatch):
    calls = _spy_on_search_and_draw(monkeypatch)

    lc_equivalence.heuristic_example2()

    assert len(calls) == 2
    first_edges = set(map(frozenset, calls[0].edges()))
    second_edges = set(map(frozenset, calls[1].edges()))
    assert first_edges != second_edges
    # Edge count alone pins the class of the second searched graph, not just its distinctness
    # from the first: the unfixed heuristic pick (index 0, i.e. K(3, 3) itself) complemented at
    # node 3 has 12 edges; the fixed pick complemented at node 0 undoes the heuristic's
    # complementation and returns K(3, 3) itself, 9 edges; only the fixed pick complemented at
    # node 3 gives this 8-edge graph.
    assert calls[1].number_of_edges() == 8


def test_heuristic_example3_searches_a_distinct_second_graph(monkeypatch):
    calls = _spy_on_search_and_draw(monkeypatch)

    lc_equivalence.heuristic_example3()

    assert len(calls) == 2
    first_edges = set(map(frozenset, calls[0].edges()))
    second_edges = set(map(frozenset, calls[1].edges()))
    assert first_edges != second_edges


def test_first_distinct_graph_raises_on_k1():
    graph = nx.complete_graph(1)
    candidates = get_lc_graph_by_max_edge(graph, 3, graph_metric_lists[0])

    with pytest.raises(ValueError):
        lc_equivalence._first_distinct_graph(candidates, graph)


def test_first_distinct_graph_raises_on_k2():
    graph = nx.complete_graph(2)
    candidates = get_lc_graph_by_max_edge(graph, 3, graph_metric_lists[0])

    with pytest.raises(ValueError):
        lc_equivalence._first_distinct_graph(candidates, graph)


def test_first_distinct_graph_does_not_raise_on_k3():
    graph = nx.complete_graph(3)
    candidates = get_lc_graph_by_max_edge(graph, 3, graph_metric_lists[0])

    result = lc_equivalence._first_distinct_graph(candidates, graph)

    assert isinstance(result, nx.Graph)
    reference_edges = set(map(frozenset, graph.edges()))
    result_edges = set(map(frozenset, result.edges()))
    assert result_edges != reference_edges


@pytest.mark.parametrize(
    ("demo", "args", "expected"),
    [
        (lc_equivalence.linear_4_example, (), 3),
        (lc_equivalence.square_4_example, (), 4),
        (lc_equivalence.repeater_graph_example, (4,), 2),
        (lc_equivalence.star_graph_example, (7,), 2),
        (lc_equivalence.biclique_example, (3,), 3),
    ],
    ids=[
        "linear_4_example",
        "square_4_example",
        "repeater_graph_example",
        "star_graph_example",
        "biclique_example",
    ],
)
def test_run_multiple_graphs_demos_search_distinct_graphs(
    demo, args, expected, monkeypatch
):
    calls = _spy_on_search_and_draw(monkeypatch)

    demo(*args)

    assert len(calls) == expected
    edge_sets = {frozenset(map(frozenset, g.edges())) for g in calls}
    assert len(edge_sets) == len(calls)
