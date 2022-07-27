import networkx as nx
import numpy as np
from matplotlib import pyplot as plt

from src.backends.stabilizer.functions.matrix_functions import rref


def height_func_list(x_matrix, z_matrix):
    """
    Calculates the height_function for all qubit in the graph given the stabilizer tableau of a graph state with ordered
    nodes. Node ordering should correspond to the rows present in the adjacency matrix. (i-th node must be i-th row)

    :param x_matrix: x matrix in the symplectic representation of the stabilizers
    :type x_matrix: np.ndarray
    :param z_matrix: z matrix in the symplectic representation of the stabilizers
    :type z_matrix: np.ndarray
    :return: the height as a function of qubit positions in graph. This is related to the entanglement entropy with
    respect to the bi-partition of the state at the given position.
    :rtype: int
    """
    n_qubits = np.shape(x_matrix)[0]
    r_vector = np.zeros([n_qubits, 1])
    height_list = []

    x_matrix, z_matrix, r_vector = rref(x_matrix, z_matrix, r_vector)

    for qubit_position in range(n_qubits):
        left_most_nontrivial = []
        for row_i in range(n_qubits):
            for column_j in range(n_qubits):
                if not (
                    x_matrix[row_i, column_j] == 0 and z_matrix[row_i, column_j] == 0
                ):
                    left_most_nontrivial.append(column_j)
                    break
        assert len(left_most_nontrivial) == n_qubits, (
            "Invalid input. One of the stabilizers is identity on " "all qubits!"
        )
        n_non_trivial_generators = len(
            [x for x in left_most_nontrivial if x - qubit_position > 0]
        )
        height = n_qubits - (qubit_position + 1) - n_non_trivial_generators
        height_list.append(height)
    return height_list


def height_function(x_matrix, z_matrix, qubit_position):
    """
    Calculates the height_function for the desired qubit in the graph given the label (position) of the qubit/node.

    :param x_matrix: x matrix in the symplectic representation of the stabilizers
    :type x_matrix: np.ndarray
    :param z_matrix: z matrix in the symplectic representation of the stabilizers
    :type z_matrix: np.ndarray
    :param qubit_position: label or position of the qubit/node in the graph
    :type qubit_position: int
    :return: the height function at the given qubit. This is related to the entanglement entropy with respect to the
    bi-partition of the state at the given position.
    :rtype: int
    """

    height = height_func_list(x_matrix, z_matrix)[qubit_position]
    return height


def height_dict(x_matrix=None, z_matrix=None, graph=None):
    """
    Generates the height_function dictionary for all qubits, given the x and z matrices or the graph the state
    corresponds to.

    :param x_matrix: x matrix in the symplectic representation of the stabilizers
    :type x_matrix: np.ndarray
    :param z_matrix: z matrix in the symplectic representation of the stabilizers
    :type z_matrix: np.ndarray
    :param graph: the graph corresponding to the state
    :type graph: networkx.classes.graph.Graph
    :return: the value of the height function for all positions in a dictionary.
    :rtype: dict
    """
    if x_matrix is None or z_matrix is None:
        if isinstance(graph, nx.classes.graph.Graph):
            n_qubits = len(graph)
            node_list = list(graph.nodes()).sort()
            # nodelist is an essential kwarg in converting graph to adjacency matrix.
            z_matrix = nx.to_numpy_array(graph, nodelist=node_list)
            x_matrix = np.eye(n_qubits)
        elif graph:
            raise ValueError("graph should be a valid networkx graph object")
        else:
            raise ValueError(
                "Either a graph or both x AND z matrices must be provided."
            )

    n_qubits = np.shape(x_matrix)[0]
    positions = [-1] + [*range(n_qubits)]
    # the first element of qubit positions list is set to -1 for symmetric plotting of the height function.
    height_x = [0] + height_func_list(x_matrix, z_matrix)
    # the first element of height function is set to zero and corresponds to an imaginary qubit at position -1.

    h_dict = {positions[i]: height_x[i] for i in range(n_qubits + 1)}
    return h_dict


def height_max(x_matrix=None, z_matrix=None, graph=None):
    """
    Given the x and z matrices or the graph the state corresponds to. Returns the maximum of height function which is
    equal to the minimum number of emitter needed for deterministic generation of the state

    :param x_matrix: x matrix in the symplectic representation of the stabilizers
    :type x_matrix: np.ndarray
    :param z_matrix: z matrix in the symplectic representation of the stabilizers
    :type z_matrix: np.ndarray
    :param graph: the graph corresponding to the state
    :type graph: networkx.classes.graph.Graph
    :return: maximum of height function over all qubits.
    :rtype: int
    """
    h_dict = height_dict(x_matrix=x_matrix, z_matrix=z_matrix, graph=graph)
    h_max = h_dict[max(h_dict, key=h_dict.get)]
    return h_max


def height_plotter(h_dict):
    """
    Plots the height function.

    :param h_dict: the height function dict which is the output of the ``height_dict``.
    :type h_dict: dict
    :return: maximum of height function over all qubits.
    :rtype: int
    """
    h_max = h_dict[max(h_dict, key=h_dict.get)]
    positions = list(h_dict.keys())
    height_x = list(h_dict.values())
    number_of_qubits = len(positions) - 1
    fig1, ax1 = plt.subplots(1, 1, constrained_layout=True, sharey=True)
    ax1.plot(positions, height_x, marker="o", markerfacecolor="red", markersize=8)
    ax1.set_title("The height function")
    ax1.set_xlabel("qubit position")
    ax1.set_ylabel("Bipartite Entanglement")
    ax1.set(xlim=(-1, number_of_qubits - 1), ylim=(0, h_max + 1))
    ax1.set_yticks(range(0, h_max + 1))
    ax1.set_xticks(positions)
    plt.show()