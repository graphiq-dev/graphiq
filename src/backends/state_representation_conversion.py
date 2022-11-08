"""
State representation conversion module. This conversion module uses the internal representations instead of classes from
src.backends.*.states

"""

import numpy as np
import networkx as nx

import src.backends.density_matrix.functions as dmf
import src.backends.graph.functions as gf
import src.backends.stabilizer.functions.utils as sfu
import src.backends.stabilizer.functions.linalg as slinalg

# TODO: Currently the conversion functions assume no redundant encoding. Next step is to include redundant encoding.
# TODO: We currently assume exact state conversion, that is, no need to check local-Clifford equivalency. Need to be
#       able to convert local-Clifford equivalent states.


def _graph_finder(x_matrix, z_matrix):
    """
    A helper function to obtain the (closest) local Clifford-equivalent graph to the stabilizer representation The
    local Clifford equivalency needs to be checked via the stabilizer of the resulting graph and the initial stabilizer

    :param x_matrix: binary matrix for representing Pauli X part of the symplectic binary
        representation of the stabilizer generators
    :type x_matrix: numpy.ndarray
    :param z_matrix:binary matrix for representing Pauli Z part of the
        symplectic binary representation of the stabilizer generators
    :type z_matrix: numpy.ndarray
    :raises AssertionError: if stabilizer generators are not independent,
        or if the final X part is not the identity matrix
    :return: a networkx.Graph object that represents the graph corresponding to the stabilizer
    :rtype: networkX.Graph
    """
    n_row, n_column = x_matrix.shape
    x_matrix, z_matrix, rank = slinalg.row_reduction(x_matrix, z_matrix)

    if x_matrix[rank][n_column - 1] == 0:
        rank = rank - 1
    positions = [*range(rank + 1, n_row)]

    x_matrix, z_matrix = slinalg.hadamard_transform(x_matrix, z_matrix, positions)

    assert (
        np.linalg.det(x_matrix)
    ) % 2 != 0, "Stabilizer generators are not independent."
    x_inverse = np.linalg.inv(x_matrix)
    x_matrix, z_matrix = (
        np.matmul(x_inverse, x_matrix) % 2,
        np.matmul(x_inverse, z_matrix) % 2,
    )

    # remove diagonal parts of z_matrix
    z_diag = list(np.diag(z_matrix))

    for i in range(len(z_diag)):
        if z_diag[i] == 1:
            z_matrix[i, i] = 0

    assert (x_matrix.shape[0] == x_matrix.shape[1]) and (
        np.allclose(x_matrix, np.eye(x_matrix.shape[0]))
    ), "Unexpected X matrix."

    state_graph = nx.from_numpy_array(z_matrix)
    return state_graph


def density_to_graph(input_matrix, threshold=0.1):
    """
    Converts a density matrix state representation to a graph representation via an adjacency matrix
    It assumes qubit systems.

    :param input_matrix: a density matrix to be converted to a graph
    :type input_matrix: numpy.ndarray
    :param threshold: a minimum threshold value of negativity for the state to assume entangled
    :type threshold: double or float
    :return: an adjacency matrix representation
    :rtype: numpy.ndarray
    """

    if isinstance(input_matrix, np.ndarray):
        rho = input_matrix
    else:
        raise TypeError("Input density matrix must be a numpy.ndarray")
    n_qubits = int(np.log2(np.sqrt(rho.size)))
    graph_adj = np.zeros((n_qubits, n_qubits))
    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            measurement_locations = [
                1 if (k is not i) and (k is not j) else 0 for k in range(n_qubits)
            ]

            rho_ij = dmf.project_to_z0_and_remove(rho, measurement_locations)
            neg_ij = dmf.negativity(rho_ij, 2, 2)
            if neg_ij > threshold:
                graph_adj[i, j] = 1

    # turn the upper triangular matrix to the adjacency matrix
    graph_adj = graph_adj + graph_adj.T
    return graph_adj


def graph_to_density(input_graph):
    """
    Builds a density matrix representation from a graph (either networkx.graph or a Graph representation)

    :param input_graph: the graph from which we will build a density matrix
    :type input_graph: networkx.Graph
    :raise TypeError: if input_graph is not of the type of networkx.Graph or a src.backends.graph.state.Graph
        or an adjacency matrix given by a numpy array
    :return: a DensityMatrix representation with the data contained by graph
    :rtype: DensityMatrix
    """

    if isinstance(input_graph, nx.Graph):
        graph_data = input_graph
    elif isinstance(input_graph, np.ndarray):
        graph_data = nx.from_numpy_array(input_graph)
    else:
        raise TypeError(
            "Input graph must be NetworkX graph or adjacency matrix using numpy.array."
        )

    number_qubits = graph_data.number_of_nodes()
    mapping = dict(zip(graph_data.nodes(), range(0, number_qubits)))
    edge_list = list(graph_data.edges)
    final_state = dmf.create_n_plus_state(number_qubits)

    for edge in edge_list:
        cz = dmf.get_two_qubit_controlled_gate(number_qubits, mapping[edge[0]], mapping[edge[1]], dmf.sigmaz())
        final_state = cz @ final_state @ np.conjugate(cz.T)

    return final_state


def graph_to_stabilizer(input_graph):
    """
    Convert a graph to stabilizer
    :param input_graph: the input graph to be converted to the stabilizer
    :type input_graph: networkX.Graph or numpy.ndarray
    :raise TypeError: if input_graph is not of the type of networkx.Graph or a src.backends.graph.state.Graph
        or an adjacency matrix given by a numpy array
    :return: two binary matrices representing the stabilizer generators
    :rtype: np.ndarray
    """
    if isinstance(input_graph, nx.Graph):
        adj_matrix = nx.to_numpy_array(input_graph)
    elif isinstance(input_graph, np.ndarray):
        adj_matrix = input_graph
    else:
        raise TypeError(
            "Input graph must be NetworkX graph or adjacency matrix using numpy.array."
        )
    n_nodes = int(np.sqrt(adj_matrix.size))

    return sfu.symplectic_to_string(np.eye(n_nodes), adj_matrix)


def stabilizer_to_graph(input_stabilizer):
    """
    Convert a stabilizer to graph

    :param input_stabilizer: the stabilizer representation in terms of a list of strings
    :type input_stabilizer: list[str]
    :return: a graph representation
    :rtype: networkX.Graph
    """

    x_matrix, z_matrix = sfu.string_to_symplectic(input_stabilizer)
    graph = _graph_finder(x_matrix, z_matrix)
    return graph


def stabilizer_to_density(input_stabilizer):
    """
    Convert a stabilizer to a density matrix

    :param input_stabilizer: the stabilizer representation in terms of a list of strings
    :type input_stabilizer: list[str]
    :return: a density matrix
    :rtype: numpy.ndarray
    """
    n_generators = len(input_stabilizer)
    n_qubits = len(input_stabilizer[0])
    assert n_generators == n_qubits
    rho = np.eye(2**n_qubits)
    for generator in input_stabilizer:
        stabilizer_elem = sfu.get_stabilizer_element_by_string(generator)
        rho = np.matmul(rho, (stabilizer_elem + np.eye(2**n_qubits)) / 2)

    return rho


def density_to_stabilizer(input_matrix):
    """
    Convert a density matrix to stabilizer via graph representation

    This works only for graph states, not for general stabilizer states that are not graph states
    TODO: generalize to general stabilizer states

    :param input_matrix: a density matrix
    :type input_matrix: numpy.ndarray
    :return: a stabilizer representation
    :rtype: list[str]
    """
    graph = density_to_graph(input_matrix)
    stabilizer = graph_to_stabilizer(graph)
    return stabilizer
