"""
This module helps with the creation of various graph states (which we can use as targets for solving)

It offers various helper functions to create graphs of various sizes. Nodes are not encoded with any redundancy at
this point (not necessary for our current algorithms)
"""
import networkx as nx
import copy
import numpy as np
import matplotlib.pyplot as plt

from src.backends.graph.state import Graph
from src.backends.lc_equivalence_check import is_lc_equivalent


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
    return Graph(nx.star_graph(points), 1)


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
    :return: a list of graphs LC equivalent to start graph, then a list of numpy arrays detailing the list of
             complementations used for each graph
    :rtype: list[Graphs], list[numpy.ndarray]
    """
    lc_graphs = []
    lc_sequences = []
    if rng is None:
        rng = np.random.default_rng()

    seq_lengths = rng.integers(low=1, high=max_seq_length + 1, size=n_graphs)

    for seq_length in seq_lengths:
        new_graph = copy.deepcopy(start_graph)
        nodes = rng.choice(new_graph.get_nodes_id_form(), size=seq_length)
        for node in nodes:
            new_graph.local_complementation(node)
        lc_graphs.append(new_graph)
        lc_sequences.append(nodes)

    return lc_graphs, lc_sequences


if __name__ == "__main__":
    np.random.seed(1)  # for consistent graph visualization
    g1 = lattice_cluster_state((2, 3))
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_title("g1")
    g1.draw(ax=ax)

    g2 = copy.deepcopy(g1)
    g2.local_complementation(2)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_title("g2")
    g2.draw(ax=ax)

    g3 = copy.deepcopy(g2)
    g3.local_complementation(1)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_title("g3")
    g3.draw(ax=ax)

    equiv12, _ = g1.lc_equivalent(g2, mode="deterministic")
    equiv13, _ = g1.lc_equivalent(g3, mode="deterministic")
    equiv23, _ = g2.lc_equivalent(g3, mode="deterministic")

    print(f"g1, g2 LC equivalent: {equiv12}")
    print(f"g1, g3 LC equivalent: {equiv13}")
    print(f"g2, g3 LC equivalent: {equiv23}")
