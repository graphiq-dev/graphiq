"""
This module helps with the creation of various graph states (which we can use as targets for solving)

It offers various helper functions to create graphs of various sizes. Nodes are not encoded with any redundancy at
this point (not necessary for our current algorithms)
"""
import networkx as nx
import copy
import numpy as np

from src.backends.graph.state import Graph


def linear_cluster_state(n_qubits):
    """
    Generates a linear cluster state

    :param n_qubits: the number of qubits in the cluster state
    :type n_qubits: int
    :return: a linear cluster state graph
    :rtype: Graph
    """
    edge_tuples = [(i, i + 1) for i in range(1, n_qubits)]
    return Graph(nx.Graph(edge_tuples), 1)


def lattice_cluster_state(dimension):
    """
    Generates an n_side by n_side cluster state lattice

    :param dimension: a tuple containing the number of qubits along each dimension
    :type dimension: tuple
    :return: a lattice cluster state
    :rtype: Graph
    """
    g = nx.convert_node_labels_to_integers(nx.grid_graph(dim=dimension), first_label=1)
    return Graph(g, 1)


def star_state(points):
    """
    Creates a start state with points qubits attached to the central qubit

    :param points: the number of nodes attached to the central node
    :type points: int
    :return: star-shaped graph state
    :rtype: Graph
    """
    edge_tuples = [(1, i) for i in range(2, 2 + points)]
    return Graph(nx.Graph(edge_tuples), 1)


def random_lc_equivalent(start_graph, n_graphs, max_seq_length, rng=None):
    """
    Random generates n_graphs new graphs from start_graphs, via local complementation.
    The maximum number of local complementations used to generate the new graph is max_seq_length
    This function does NOT enforce that all graphs must be different

    :param start_graph: the initial graph from which the other LC equivalent graphs are made
    :param n_graphs: the number of random graphs generated
    :type n_graphs: int
    :param max_seq_length: the maximum number of local complementations applied to generate the new graph
    :type max_seq_length: int
    :return: a list of graphs LC equivalent to start graph
    :rtype: list[Graphs]
    """
    lc_graphs = []
    if rng is None:
        rng = np.random.default_rng()

    seq_lengths = rng.integers(low=1, high=max_seq_length + 1, size=n_graphs)

    for seq_length in seq_lengths:
        new_graph = copy.deepcopy(start_graph)
        for _ in range(seq_length):
            node = rng.choice(new_graph.get_nodes_id_form())
            new_graph.local_complementation(node)
        lc_graphs.append(new_graph)

    return lc_graphs


if __name__ == "__main__":
    rng = np.random.default_rng(seed=0)
    start_graph_test = lattice_cluster_state((3, 3))
    lc_graphs_test = random_lc_equivalent(start_graph_test, 2, 3)
    for lc_graph in lc_graphs_test:
        lc_graph.draw()