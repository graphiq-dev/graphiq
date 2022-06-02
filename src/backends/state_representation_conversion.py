import numpy as np
import networkx as nx

import src.backends.density_matrix.functions as dmf
import src.backends.graph.functions as gf
import src.backends.stabilizer.functions as sf

from src.backends.density_matrix.state import DensityMatrix
from src.backends.graph.state import Graph
from src.backends.stabilizer.state import Stabilizer


# TODO: Currently the conversion functions assume no redundant encoding. Next step is to include redundant encoding.

def _graph_finder(x_matrix, z_matrix, pivot):
    """
    The overall function getting X and Z matrices as input and giving out the equivalent graph. Initialization of X,
    Z, and pivot is needed.

    :param x_matrix: binary matrix for representing Pauli X part of the symplectic binary
    representation of the stabilizer generators
    :param z_matrix:binary matrix for representing Pauli Z part of the
    symplectic binary representation of the stabilizer generators
    :param pivot: a location to start
    :return: a networkx.Graph object
    """
    n, m = np.shape(x_matrix)
    x_matrix, z_matrix, rank = sf.row_reduction(x_matrix, z_matrix, pivot)
    if x_matrix[rank][np.shape(x_matrix)[1] - 1] == 0:
        rank = rank - 1
    positions = [*range(rank + 1, n)]
    x_matrix, z_matrix = sf.hadamard_transform(x_matrix, z_matrix, positions)
    assert ((np.linalg.det(x_matrix)) % 2 != 0), "Stabilizer generators are not independent!"
    x_inverse = np.linalg.inv(x_matrix)
    x_matrix, z_matrix = np.matmul(x_inverse, x_matrix) % 2, np.matmul(x_inverse, z_matrix) % 2

    # remove diagonal parts of z_matrix
    z_diag = list(np.diag(z_matrix))

    indices = []
    for i in range(len(z_diag)):
        indices.append(i)

    for i in indices:
        z_matrix[i, i] = 0

    assert (x_matrix.shape[0] == x_matrix.shape[1]) and (
            x_matrix == np.eye(x_matrix.shape[0])).all(), "something is wrong!"

    # print("Final X matrix", "\n", x_matrix, "\n")
    # print("Final Z matrix", "\n", z_matrix, "\n")
    state_graph = nx.from_numpy_matrix(z_matrix)
    return state_graph


def density_to_graph(input_matrix, threshold=0.1):
    """
    Converts a density matrix state representation to a graph representation via an adjacency matrix
    It assumes qubit systems.

    :param input_matrix:
    :type input_matrix: numpy.ndarray
    :param threshold: a minimum threshold value of negativity for the state to assume entangled
    :type threshold: double or float
    :return: an adjacency matrix representation
    :rtype: numpy.ndarray
    """

    n_qubits = int(np.log2(np.sqrt(input_matrix.size)))
    graph_adj = np.zeros((n_qubits, n_qubits))
    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            measurement_locations = [1 if (k is not i) and (k is not j) else 0 for k in range(n_qubits)]

            rho_ij = dmf.project_to_z0_and_remove(input_matrix, measurement_locations)
            neg_ij = dmf.negativity(rho_ij, 2, 2)
            if neg_ij > threshold:
                graph_adj[i, j] = 1
    graph_adj = graph_adj + graph_adj.T
    return graph_adj


def graph_to_density(input_graph):
    """
    Builds a density matrix representation from a graph (either nx.graph or a Graph state representation)

    :param input_graph: the graph from which we will build a density matrix
    :type input_graph: networkx.Graph OR Graph
    :return: a DensityMatrix representation with the data contained by graph
    :rtype: DensityMatrix
    """

    if isinstance(input_graph, nx.Graph):
        graph_data = input_graph
    elif isinstance(input_graph, Graph):
        graph_data = input_graph.data
    else:
        raise TypeError("Input graph must be Graph object or NetworkX graph.")

    number_qubits = graph_data.number_of_nodes()
    mapping = dict(zip(graph_data.nodes(), range(0, number_qubits)))
    edge_list = list(graph_data.edges)
    final_state = dmf.create_n_plus_state(number_qubits)

    for edge in edge_list:
        final_state = dmf.apply_cz(final_state, mapping[edge[0]], mapping[edge[1]])

    return final_state


def graph_to_stabilizer(input_graph):
    """
    Convert a graph to stabilizer
    """
    # TODO:
    raise NotImplementedError('Next step')


def stabilizer_to_graph(input_stabilizer):
    """
    Convert a stabilizer to graph

    """

    x_matrix, z_matrix = sf.stabilizer_to_symplectic(input_stabilizer)

    pivot = [0, 0]
    graph = _graph_finder(x_matrix, z_matrix, pivot)
    return graph


def stabilizer_to_density(input_stabilizer):
    """
    Convert a stabilizer to a density matrix
    """
    raise NotImplementedError('Next step')


def density_to_stabilizer(input_matrix):
    """
    Convert a density matrix to stabilizer

    """
    graph = density_to_graph(input_matrix)
    stabilizer = graph_to_stabilizer(graph)
    return stabilizer
