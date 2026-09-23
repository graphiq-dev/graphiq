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
    graph = nx.Graph([(0, 1), (1, 2), (2, 3)])
    run_multiple_graphs_and_report(graph, [1, 2])


def square_4_example():
    graph = nx.Graph([(0, 1), (0, 2), (2, 3), (1, 3), (1, 2)])
    # nodes 0 and 3 share the neighbourhood {1, 2}, so LC(0) == LC(3)
    run_multiple_graphs_and_report(graph, [0, 1, 2])


def repeater_graph_example(n_inner_qubits):
    graph = repeater_graph_states(n_inner_qubits)
    run_multiple_graphs_and_report(graph, [1])


def star_graph_example(n_points):
    graph = nx.star_graph(n_points)
    # LC at a degree-1 leaf is the identity; only the centre changes the graph
    run_multiple_graphs_and_report(graph, [0])


def biclique_example(n_pairs):
    # built by the generator so nodes are inserted in ascending order: the local
    # complementation indexes nodes by adjacency-matrix position, and an edge
    # list inserts them as 0, n_pairs, ... and complements the wrong ones.
    # Nodes on one side share a neighbourhood, so LC(0) == LC(1).
    graph = nx.complete_bipartite_graph(n_pairs, n_pairs)
    run_multiple_graphs_and_report(graph, [0, n_pairs])


def imperfect_repeater_graph_example():
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
