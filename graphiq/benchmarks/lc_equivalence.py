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
Demo scripts that build a graph, draw it and graphs reached from it by local
complementation, and print an alternate-circuit search report for each graph drawn. Run this
module directly to execute whichever example is left uncommented under ``__main__``.
"""

import matplotlib.pyplot as plt
from graphiq.benchmarks.alternate_circuits import (
    exemplary_run,
    report_alternate_circuits,
)
from graphiq.benchmarks.graph_states import repeater_graph_states
from graphiq.utils.preprocessing import *


def _first_distinct_graph(candidates, graph):
    """
    Return the first candidate graph whose edge set differs from ``graph``'s.

    The LC heuristics in ``graphiq.utils.preprocessing`` seed their candidate list
    with the input graph, so when the input ties on the metric (e.g. K(3, 3))
    ``candidates[0][1]`` is the input itself.

    :param candidates: (metric value, graph) pairs, lowest metric value first
    :type candidates: list
    :param graph: the graph the candidates were generated from
    :type graph: networkx.Graph
    :raises ValueError: if no candidate's edge set differs from ``graph``'s
    :return: the first candidate whose edge set differs from ``graph``'s
    :rtype: networkx.Graph
    """
    reference = set(map(frozenset, graph.edges()))
    for _, candidate in candidates:
        if set(map(frozenset, candidate.edges())) != reference:
            return candidate
    raise ValueError("no candidate graph differs from the input graph")


def run_multiple_graphs_and_report(graph, lc_nodes):
    """
    Draw the given graph and, for each node in lc_nodes, the graph reached from it by one local
    complementation at that node, and print an alternate-circuit search report for each, so
    ``1 + len(lc_nodes)`` searches and reports in all.

    Each local complementation is computed from the original graph, not cumulatively, so
    lc_nodes describes a fan out from the one graph rather than a chain. Every graph drawn is
    the graph its accompanying report describes.

    :param graph: the graph to draw and search
    :type graph: networkX.Graph
    :param lc_nodes: nodes to apply a local complementation to, one at a time
    :type lc_nodes: list
    :return: nothing
    :rtype: None
    """
    results = [exemplary_run(graph, None)]
    graphs = [graph]
    for node in lc_nodes:
        alt_graph = local_comp_graph(graph, node)

        graphs.append(alt_graph)
        results.append(exemplary_run(alt_graph, None))
    for i in range(len(results)):
        ax = plt.axes()
        nx.draw(graphs[i], ax=ax, with_labels=True)
        report_alternate_circuits(results[i])


def linear_4_example():
    """
    Run and report on a 4-node linear cluster graph (path 0-1-2-3) and the local
    complementations of it at nodes 1 and 2.

    :return: nothing
    :rtype: None
    """
    graph = nx.Graph([(0, 1), (1, 2), (2, 3)])
    run_multiple_graphs_and_report(graph, [1, 2])


def square_4_example():
    """
    Run and report on a 4-node graph (a 4-cycle 0-1-3-2-0 plus the diagonal edge 1-2) and the
    local complementations of it at nodes 0, 1 and 2.

    :return: nothing
    :rtype: None
    """
    graph = nx.Graph([(0, 1), (0, 2), (2, 3), (1, 3), (1, 2)])
    # nodes 0 and 3 share the neighbourhood {1, 2}, so LC(0) == LC(3)
    run_multiple_graphs_and_report(graph, [0, 1, 2])


def repeater_graph_example(n_inner_qubits):
    """
    Run and report on a repeater graph state with n_inner_qubits inner qubits, and the local
    complementation of it at node 1.

    :param n_inner_qubits: number of qubits in the repeater graph's inner layer
    :type n_inner_qubits: int
    :return: nothing
    :rtype: None
    """
    graph = repeater_graph_states(n_inner_qubits)
    run_multiple_graphs_and_report(graph, [1])


def star_graph_example(n_points):
    """
    Run and report on a star graph with n_points leaves attached to the central node, and the
    local complementation of it at the central node (0).

    :param n_points: number of leaf nodes attached to the central node
    :type n_points: int
    :return: nothing
    :rtype: None
    """
    graph = nx.star_graph(n_points)
    # LC at a degree-1 leaf is the identity; only the centre changes the graph
    run_multiple_graphs_and_report(graph, [0])


def biclique_example(n_pairs):
    """
    Run and report on the complete bipartite graph K(n_pairs, n_pairs) (nodes 0..n_pairs-1 on one
    side, n_pairs..2*n_pairs-1 on the other, every cross edge present), and the local
    complementations of it at nodes 0 and n_pairs.

    :param n_pairs: number of nodes on each side of the bipartite graph
    :type n_pairs: int
    :return: nothing
    :rtype: None
    """
    # built by the generator so nodes are inserted in ascending order: the local
    # complementation indexes nodes by adjacency-matrix position, and an edge
    # list inserts them as 0, n_pairs, ... and complements the wrong ones.
    # Nodes on one side share a neighbourhood, so LC(0) == LC(1).
    graph = nx.complete_bipartite_graph(n_pairs, n_pairs)
    run_multiple_graphs_and_report(graph, [0, n_pairs])


def imperfect_repeater_graph_example():
    """
    Draw and report on two 11-node graphs that share a hub node (2) linked to five inner nodes
    (1, 4, 6, 8, 10), each inner node carrying one pendant leaf (0, 3, 5, 7, 9 respectively): the
    first also has a clique on the five inner nodes, the second omits it. Neither is a repeater
    graph state, which has an even number of nodes.

    :return: nothing
    :rtype: None
    """
    edges_set1 = [(2, 1), (2, 4), (2, 6), (2, 8), (2, 10)]
    edges_set1 += [(0, 1), (3, 4), (5, 6), (7, 8), (9, 10)]
    edges_set2 = edges_set1.copy()
    node_set = [1, 4, 6, 8, 10]
    for i in range(len(node_set)):
        for j in range(i + 1, len(node_set)):
            edges_set1.append((node_set[i], node_set[j]))
    graph1 = nx.Graph(edges_set1)
    nx.draw(graph1, with_labels=True)
    result = exemplary_run(graph1, None)
    report_alternate_circuits(result)
    graph2 = nx.Graph(edges_set2)
    nx.draw(graph2, with_labels=True)
    result = exemplary_run(graph2, None)
    report_alternate_circuits(result)


def heuristic_example1():
    """
    Draw and report on the complete graph on 5 nodes, then on a graph reached from it by the
    max-edge local-complementation heuristic.

    The heuristic returns (metric value, graph) pairs, lowest metric value first, and the graph
    used here is the first one differing from the input (see ``_first_distinct_graph``). All ten
    retained candidates score -2 on ``graph_max_eccentricity``, so which one is drawn carries no
    meaning beyond being one of the lowest-scoring set.

    :return: nothing
    :rtype: None
    """
    graph = nx.complete_graph(5)
    nx.draw(graph, with_labels=True)
    result = exemplary_run(graph, None)
    report_alternate_circuits(result)

    graph2 = _first_distinct_graph(
        get_lc_graph_by_max_edge(graph, 10, graph_metric_lists[0]), graph
    )
    nx.draw(graph2, with_labels=True)
    result = exemplary_run(graph2, None)
    report_alternate_circuits(result)


def heuristic_example2():
    """
    Draw and report on the complete bipartite graph K(3, 3), then on one further local
    complementation of the graph the max-edge heuristic selects -- the lines that would draw and
    search the heuristic's own graph are commented out.

    The heuristic returns (metric value, graph) pairs, lowest metric value first, and the graph
    taken from it here is the first one differing from K(3, 3) itself (see
    ``_first_distinct_graph``): K(3, 3) ties with all its candidates at -2, so pair 0 is the
    input unchanged. That graph is ``LC(K(3, 3), 0)``, so the further complementation is at
    node 3; complementing at node 0 again would return K(3, 3) itself.

    :return: nothing
    :rtype: None
    """
    graph = nx.complete_bipartite_graph(3, 3)
    nx.draw(graph, with_labels=True)
    result = exemplary_run(graph, None)
    report_alternate_circuits(result)

    graph2 = _first_distinct_graph(
        get_lc_graph_by_max_edge(graph, 3, graph_metric_lists[0]), graph
    )
    # nx.draw(graph2, with_labels=True)
    # result = exemplary_run(graph2, None)
    # report_alternate_circuits(result)
    # graph2 is LC(K(3, 3), 0); complementing at node 0, 1 or 2 again would
    # return K(3, 3) itself, while node 3 gives a distinct graph
    graph3 = local_comp_graph(graph2, 3)
    nx.draw(graph3, with_labels=True)
    result = exemplary_run(graph3, None)
    report_alternate_circuits(result)


def heuristic_example3():
    """
    Draw and report on the complete bipartite graph K(3, 3), then on a graph reached from it by
    the max-neighbor-edge local-complementation heuristic.

    The heuristic returns (metric value, graph) pairs, lowest metric value first, and the graph
    used here is the first one differing from K(3, 3) itself (see ``_first_distinct_graph``):
    K(3, 3) ties with all its candidates at -2, so pair 0 is the input unchanged. Which distinct
    candidate is used carries no meaning beyond being one of the lowest-scoring set.

    :return: nothing
    :rtype: None
    """
    graph = nx.complete_bipartite_graph(3, 3)
    nx.draw(graph, with_labels=True)
    result = exemplary_run(graph, None)
    report_alternate_circuits(result)

    # graph2 = candidate_graphs(graph, 3)
    graph2 = _first_distinct_graph(
        get_lc_graph_by_max_neighbor_edge(graph, 3, graph_metric_lists[0]), graph
    )
    nx.draw(graph2, with_labels=True)
    result = exemplary_run(graph2, None)
    report_alternate_circuits(result)


if __name__ == "__main__":
    # linear_4_example()
    # square_4_example()
    # repeater_graph_example(4)
    # star_graph_example(7)
    # biclique_example(3)
    # imperfect_repeater_graph_example()
    heuristic_example3()
