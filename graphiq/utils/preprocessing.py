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
import heapq

import networkx as nx
import numpy as np

from graphiq.backends.lc_equivalence_check import local_comp_graph
from graphiq.backends.stabilizer.compiler import StabilizerCompiler
from graphiq.backends.stabilizer.functions.rep_conversion import (
    get_clifford_tableau_from_graph,
)
from graphiq.metrics import Infidelity
from graphiq.solvers.time_reversed_solver import TimeReversedSolver
from graphiq.state import QuantumState


def graph_circuit_depth(graph):
    """
    Find a circuit that generates the graph state with TimeReversedSolver, and return the maximum depth of
    its emitter registers.

    :param graph: the graph representing the target graph state
    :type graph: nx.Graph
    :return: the maximum emitter register depth of the circuit
    :rtype: int
    """
    target_tableau = get_clifford_tableau_from_graph(graph)
    target_state = QuantumState(target_tableau, rep_type="s")
    compiler = StabilizerCompiler()
    compiler.noise_simulation = False
    compiler.measurement_determinism = 1
    metric = Infidelity(target_state)
    solver = TimeReversedSolver(target=target_state, metric=metric, compiler=compiler)
    solver.solve()
    benchmark_score, benchmark_circuit = solver.result
    depth_list = benchmark_circuit.calculate_reg_depth("e")
    return max(depth_list)


def graph_max_eccentricity(graph):
    """
    Return the negated maximum value of node eccentricity, so that a lower value corresponds to a larger
    eccentricity

    :param graph: the graph to score
    :type graph: nx.Graph
    :raises ValueError: if the graph has no nodes
    :return: the negative of the maximum node eccentricity
    :rtype: int
    """

    node_eccentricity = nx.eccentricity(graph)
    node_eccentricity_list = node_eccentricity.values()
    if len(node_eccentricity_list) == 0:
        raise ValueError(
            "The input graph is an empty graph. Its node eccentricity is not defined."
        )
    else:
        return -max(node_eccentricity_list)


def graph_max_centrality(graph):
    """
    Return the maximum of node degree centrality

    :param graph: the graph to score
    :type graph: nx.Graph
    :raises ValueError: if the graph has no nodes
    :return: the maximum degree centrality over all nodes
    :rtype: float
    """
    degree_centrality = nx.degree_centrality(graph)
    degree_centrality_list = degree_centrality.values()
    if len(degree_centrality_list) == 0:
        raise ValueError(
            "The input graph is an empty graph. Its node centrality is not defined."
        )
    else:
        return max(degree_centrality_list)


def graph_betweenness_centrality(graph):
    """
    Return the maximum of node betweenness centrality

    :param graph: the graph to score
    :type graph: nx.Graph
    :raises ValueError: if the graph has no nodes
    :return: the maximum betweenness centrality over all nodes
    :rtype: float
    """
    betweenness_centrality = nx.betweenness_centrality(graph)
    betweenness_centrality_list = betweenness_centrality.values()
    if len(betweenness_centrality_list) == 0:
        raise ValueError(
            "The input graph is an empty graph. Its node betweenness centrality is not defined."
        )
    else:
        return max(betweenness_centrality_list)


def graph_closeness_centrality(graph):
    """
    Return the maximum of node closeness centrality

    :param graph: the graph to score
    :type graph: nx.Graph
    :raises ValueError: if the graph has no nodes
    :return: the maximum closeness centrality over all nodes
    :rtype: float
    """
    closeness_centrality = nx.closeness_centrality(graph)
    closeness_centrality_list = closeness_centrality.values()
    if len(closeness_centrality_list) == 0:
        raise ValueError(
            "The input graph is an empty graph. Its node closeness centrality is not defined."
        )
    else:
        return max(closeness_centrality_list)


def _select_graphs(candidate_graphs, new_graph, limit, metric_value):
    """
    Helper function to select graphs: add new_graph to the candidates and keep
    the `limit` entries with the lowest metric value

    :param candidate_graphs: the current candidates, as (metric value, graph) pairs
    :type candidate_graphs: list[(float, networkx.Graph)]
    :param new_graph: the graph to consider adding
    :type new_graph: networkx.Graph
    :param limit: the maximum number of candidates to keep
    :type limit: int
    :param metric_value: the metric value of new_graph
    :type metric_value: float
    :return: the updated candidates, sorted by ascending metric value
    :rtype: list[(float, networkx.Graph)]
    """
    # keep the `limit` lowest-metric entries; the list was previously only
    # sorted once full, so the best candidate could be evicted. key= avoids
    # comparing the graphs themselves on ties.
    return heapq.nsmallest(
        limit,
        candidate_graphs + [(metric_value, new_graph)],
        key=lambda pair: pair[0],
    )


def get_lc_graph_by_max_edge(graph, n_graphs, graph_metric, n_trial=5):
    """
    Search the local-complementation orbit of a graph: for n_trial rounds, locally complement each candidate
    graph at every node with the maximum number of edges (degree), score each resulting graph with graph_metric,
    and keep the n_graphs candidates with the lowest score.

    :param graph: the graph to search from
    :type graph: nx.Graph or numpy.ndarray
    :param n_graphs: the maximum number of candidate graphs to keep
    :type n_graphs: int
    :param graph_metric: a function scoring a graph, called as graph_metric(graph)
    :type graph_metric: function
    :param n_trial: the number of local-complementation rounds; defaults to 5
    :type n_trial: int
    :raises TypeError: if the type of graph is not exactly nx.Graph or numpy.ndarray (subclasses such as
        nx.DiGraph or nx.MultiGraph are rejected)
    :return: at most n_graphs (score, graph) tuples, sorted by ascending score
    :rtype: list[(float, nx.Graph)]
    """
    if type(graph) == np.ndarray:
        graph = nx.from_numpy_array(graph)
    elif type(graph) == nx.Graph:
        pass
    else:
        raise TypeError(f"The type {type(graph)} is not supported.")
    score = graph_metric(graph)
    candidate_graphs = [(score, graph)]

    for i in range(n_trial):
        tmp_graphs = []
        for score, each_graph in candidate_graphs:
            adj_matrix = nx.to_numpy_array(each_graph)
            nonzero_counts = np.count_nonzero(adj_matrix, axis=0)
            max_node_ids = np.argwhere(nonzero_counts == np.amax(nonzero_counts))
            for node_id in max_node_ids:
                tmp_graphs.append(local_comp_graph(each_graph, node_id))

        for tmp_graph in tmp_graphs:
            score = graph_metric(tmp_graph)
            candidate_graphs = _select_graphs(
                candidate_graphs, tmp_graph, n_graphs, score
            )
    return candidate_graphs


def _count_n_neighbor_edges(adj_matrix):
    n_node = adj_matrix.shape[0]
    edges_count_list = np.zeros(n_node)
    for index in range(n_node):
        neighbor_list = np.nonzero(adj_matrix[index])[0]
        sub_adj_matrix = adj_matrix[np.ix_(neighbor_list, neighbor_list)]
        edges_count_list[index] = int(np.count_nonzero(sub_adj_matrix) / 2)
    return edges_count_list


def _count_n_neighbor(adj_matrix):
    n_node = adj_matrix.shape[0]
    neighbor_count_list = np.zeros(n_node)
    for index in range(n_node):
        neighbor_list = np.nonzero(adj_matrix[index])[0]
        neighbor_count_list[index] = len(neighbor_list)
    return neighbor_count_list


def get_lc_graph_by_max_neighbor_edge(graph, n_graphs, graph_metric, n_trial=5):
    """
    Search the local-complementation orbit of a graph: for n_trial rounds, locally complement each candidate
    graph at every node whose neighbors have the maximum number of edges among themselves, score each resulting
    graph with graph_metric, and keep the n_graphs candidates with the lowest score.

    :param graph: the graph to search from
    :type graph: nx.Graph or numpy.ndarray
    :param n_graphs: the maximum number of candidate graphs to keep
    :type n_graphs: int
    :param graph_metric: a function scoring a graph, called as graph_metric(graph)
    :type graph_metric: function
    :param n_trial: the number of local-complementation rounds; defaults to 5
    :type n_trial: int
    :raises TypeError: if the type of graph is not exactly nx.Graph or numpy.ndarray (subclasses such as
        nx.DiGraph or nx.MultiGraph are rejected)
    :return: at most n_graphs (score, graph) tuples, sorted by ascending score
    :rtype: list[(float, nx.Graph)]
    """

    if type(graph) == np.ndarray:
        graph = nx.from_numpy_array(graph)
    elif type(graph) == nx.Graph:
        pass
    else:
        raise TypeError(f"The type {type(graph)} is not supported.")
    score = graph_metric(graph)
    candidate_graphs = [(score, graph)]

    for i in range(n_trial):
        tmp_graphs = []
        for score, each_graph in candidate_graphs:
            adj_matrix = nx.to_numpy_array(each_graph)
            edges_count_list = _count_n_neighbor_edges(adj_matrix)

            max_node_ids = np.argwhere(edges_count_list == np.amax(edges_count_list))
            for node_id in max_node_ids:
                tmp_graphs.append(local_comp_graph(each_graph, node_id))

        for tmp_graph in tmp_graphs:
            score = graph_metric(tmp_graph)
            candidate_graphs = _select_graphs(
                candidate_graphs, tmp_graph, n_graphs, score
            )

    return candidate_graphs


graph_metric_lists = [
    graph_max_eccentricity,
    graph_max_centrality,
    graph_betweenness_centrality,
    graph_closeness_centrality,
]
