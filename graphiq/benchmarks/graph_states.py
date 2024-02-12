"""
This module helps with the creation of various graph states (which we can use as targets for solving)

It offers various helper functions to create graphs of various sizes. Nodes are not encoded with any redundancy at
this point (not necessary for our current algorithms)
"""

import networkx as nx
import copy
import numpy as np
import matplotlib.pyplot as plt
import itertools

from graphiq.backends.graph.state import Graph


def linear_cluster_state(n_qubits):
    """
    Generates a linear cluster state

    :param n_qubits: the number of qubits in the cluster state
    :type n_qubits: int
    :return: a linear cluster state graph
    :rtype: Graph
    """
    edge_tuples = [(i, i + 1) for i in range(1, n_qubits)]
    return Graph(nx.Graph(edge_tuples))


def lattice_cluster_state(dimension):
    """
    Generates an n_side by n_side cluster state lattice

    :param dimension: a tuple containing the number of qubits along each dimension
    :type dimension: tuple
    :return: a lattice cluster state
    :rtype: Graph
    """
    g = nx.convert_node_labels_to_integers(nx.grid_graph(dim=dimension), first_label=1)
    return Graph(g)


def star_graph_state(points):
    """
    Creates a start state with points qubits attached to the central qubit

    :param points: the number of nodes attached to the central node
    :type points: int
    :return: star-shaped graph state
    :rtype: Graph
    """
    return Graph(nx.star_graph(points))


def random_graph_state(n_qubits, p_edge, np_rng=None):
    """
    Creates a random graph where each edge is added with probability

    :param n_qubits: number of qubits in the graph state
    :type n_qubits: int
    :param p_edge: the probability of an edge existing between any two nodes a, b
                   this governs the sparsity of the graph
    :type p_edge: float
    :param np_rng: random number generator to use (useful to guaranteeing replicable results, when seeded)
    :type np_rng: numpy.random.Generator
    :return: a random Graph state
    :rtype: Graph
    """
    if np_rng is None:
        np_rng = np.random.default_rng()

    return Graph(_gnp_random_connected_graph(n_qubits, p_edge, np_rng))


def _gnp_random_connected_graph(n, p, np_rng):
    """
    Code modified from stackoverflow. Citation:

    yatu (2020, May 23). How to create random graph where each node has at least 1 edge using Networkx. Stack Overflow.
    Retrieved July 18, 2022, from
    https://stackoverflow.com/questions/61958360/how-to-create-random-graph-where-each-node-has-at-least-1-edge-using-networkx/61961881#61961881

    Networkx's random graph generation functions do NOT guarantee connectedness, which is something we require of our
    graph states, so, we used the suggestion above.

    :param n: the number of nodes
    :type n: int
    :param p: the probability of an edge being added
    :type p: float
    :return:
    :rtype: networkx.Graph
    """
    edges = itertools.combinations(range(n), 2)
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    if p <= 0:
        return graph
    if p >= 1:
        return nx.complete_graph(n, create_using=graph)

    for _, node_edges in itertools.groupby(edges, key=lambda x: x[0]):
        node_edges = list(node_edges)
        random_edge_index = np_rng.choice(len(node_edges))
        graph.add_edge(*node_edges[random_edge_index])
        for e in node_edges:
            if np_rng.random() < p:
                graph.add_edge(*e)
    return graph


def random_lc_equivalent(start_graph, n_graphs, max_seq_length, np_rng=None):
    """
    Random generates n_graphs new graphs from start_graphs, via local complementation.
    The maximum number of local complementations used to generate the new graph is max_seq_length
    This function does NOT enforce that all graphs must be different

    :param start_graph: the initial graph from which the other LC equivalent graphs are made
    :param n_graphs: the number of random graphs generated
    :type n_graphs: int
    :param max_seq_length: the maximum number of local complementations applied to generate the new graph
    :type max_seq_length: int
    :param np_rng: random number generator to use (useful to guaranteeing replicable results, when seeded)
    :type np_rng: numpy.random.Generator
    :return: a list of graphs LC equivalent to start graph, then a list of numpy arrays detailing the list of
             complementations used for each graph
    :rtype: list[Graphs], list[numpy.ndarray]
    """
    lc_graphs = []
    lc_sequences = []
    if np_rng is None:
        np_rng = np.random.default_rng()

    seq_lengths = np_rng.integers(low=1, high=max_seq_length + 1, size=n_graphs)

    for seq_length in seq_lengths:
        new_graph = copy.deepcopy(start_graph)
        nodes = np_rng.choice(new_graph.data, size=seq_length)
        for node in nodes:
            new_graph.local_complementation(node)
        lc_graphs.append(new_graph)
        lc_sequences.append(nodes)

    return lc_graphs, lc_sequences


def lc_equivalence_demo():
    """
    Function originally used for debugging LC equivalence. Kept as a demonstration of how graphs evolve
    under LC complementation

    :return: nothing
    :rtype: None
    """
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


def repeater_graph_states(n_inner_qubits):
    """
    Construct a repeater graph state with the emission ordering (leaf qubit -> adjacent inner qubit -> next leaf qubit)

    :param n_inner_qubits: number of qubits in the inner layer.
    :type n_inner_qubits: int
    :return: a networkx graph that represents the repeater graph state
    :rtype: networkx.Graph
    """
    edges = []
    for i in range(n_inner_qubits):
        edges.append((2 * i, 2 * i + 1))

    for i in range(n_inner_qubits):
        for j in range(i + 1, n_inner_qubits):
            edges.append((2 * i + 1, 2 * j + 1))
    return nx.Graph(edges)


def bi_repeater_graph_states(n_inner_pairs):
    """
    Construct a biclique repeater graph state with the emission ordering (leaf qubit -> adjacent inner qubit ->
    next leaf qubit)

    :param n_inner_pairs: number of pairs of qubits in the inner layer. Total number of nodes is 4n_inner_pairs.
    :type n_inner_pairs: int
    :return: a networkx graph that represents the repeater graph state
    :rtype: networkx.Graph
    """
    edges = []
    for i in range(n_inner_pairs):
        edges.append((4 * i, 4 * i + 1))
        edges.append((4 * i + 2, 4 * i + 3))

    for i in range(n_inner_pairs):
        for j in range(n_inner_pairs):
            edges.append((4 * i + 1, 4 * j + 3))
    return nx.Graph(edges)
