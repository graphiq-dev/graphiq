import networkx as nx
import numpy as np

from src.backends.lc_equivalence_check import local_comp_graph
from src.solvers.deterministic_solver import DeterministicSolver
from src.metrics import Infidelity
from src.state import QuantumState
from src.backends.stabilizer.compiler import StabilizerCompiler
from src.backends.stabilizer.functions.rep_conversion import (
    get_clifford_tableau_from_graph,
)


def graph_circuit_depth(graph):
    """
    Return max of depth of emitter registers.

    :param graph:
    :type graph:
    :return:
    :rtype:
    """
    target_tableau = get_clifford_tableau_from_graph(graph)
    n_photon = target_tableau.n_qubits
    target_state = QuantumState(n_photon, target_tableau, rep_type="stabilizer")
    compiler = StabilizerCompiler()
    compiler.noise_simulation = False
    compiler.measurement_determinism = 1
    metric = Infidelity(target_state)
    solver = DeterministicSolver(target=target_state, metric=metric, compiler=compiler)
    solver.solve()
    benchmark_score, benchmark_circuit = solver.result
    depth_list = benchmark_circuit.calculate_reg_depth("e")
    return max(depth_list)


def graph_max_eccentricity(graph):
    """
    Return the maximum value of node eccentricity

    :param graph:
    :type graph:
    :return:
    :rtype:
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

    :param graph:
    :type graph:
    :return:
    :rtype:
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

    :param graph:
    :type graph:
    :return:
    :rtype:
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

    :param graph:
    :type graph:
    :return:
    :rtype:
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
    Helper function to select graphs

    :param candidate_graphs:
    :type candidate_graphs:
    :param new_graph:
    :type new_graph:
    :param limit:
    :type limit:
    :param metric_value:
    :type metric_value:
    :return:
    :rtype:
    """
    if len(candidate_graphs) < limit:
        candidate_graphs.append((metric_value, new_graph))
    else:
        for i in range(len(candidate_graphs)):
            if metric_value < candidate_graphs[i][0]:
                candidate_graphs.insert(i, (metric_value, new_graph))
                candidate_graphs.pop()
                break
    return candidate_graphs


def get_lc_graph_by_max_edge(graph, n_graphs, graph_metric, n_trial=5):
    """

    :param graph:
    :type graph:
    :param n_graphs:
    :type n_graphs:
    :param graph_metric:
    :type graph_metric:
    :param n_trial:
    :type n_trial:
    :return:
    :rtype:
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

    :param graph:
    :type graph:
    :param n_graphs:
    :type n_graphs:
    :param graph_metric:
    :type graph_metric:
    :param n_trial:
    :type n_trial:
    :return:
    :rtype:
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
