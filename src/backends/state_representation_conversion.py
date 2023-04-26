"""
State representation conversion module. This conversion module uses the internal representations instead of classes from
src.backends.*.states

"""
import copy

import numpy as np
import networkx as nx

import src.backends.density_matrix.functions as dmf
from src.backends.density_matrix import numpy as dmnp
import src.backends.graph.functions as gf
import src.backends.stabilizer.functions.utils as sfu
import src.backends.stabilizer.functions.linalg as slinalg
from src.backends.stabilizer.functions.rep_conversion import (
    get_stabilizer_tableau_from_graph,
)
from src.backends.stabilizer.functions.stabilizer import canonical_form
from src.backends.stabilizer.functions.transformation import run_circuit
from src.backends.stabilizer.tableau import CliffordTableau, StabilizerTableau


# TODO: Currently the conversion functions assume no redundant encoding. Next step is to include redundant encoding.
# TODO: We currently assume exact state conversion, that is, no need to check local-Clifford equivalency. Need to be
#       able to convert local-Clifford equivalent states.


def _graph_finder(x_matrix, z_matrix, get_ops_data=False):
    """
    A helper function to obtain the (closest) local Clifford-equivalent graph to the stabilizer representation The
    local Clifford equivalency needs to be checked via the stabilizer of the resulting graph and the initial stabilizer

    :param x_matrix: binary matrix for representing Pauli X part of the symplectic binary
        representation of the stabilizer generators
    :type x_matrix: numpy.ndarray
    :param z_matrix:binary matrix for representing Pauli Z part of the
        symplectic binary representation of the stabilizer generators
    :type z_matrix: numpy.ndarray
    :param get_ops_data: if True, the function also returns the position (qubits) of the applied "Hadamard"
     gates and the position of the applied "P_dag" gates in a tuple.
    :type get_ops_data: bool
    :raises AssertionError: if stabilizer generators are not independent,
        or if the final X part is not the identity matrix
    :return: a networkx.Graph object that represents the graph corresponding to the stabilizer
    :rtype: networkX.Graph
    """
    x_mat = np.copy(x_matrix)
    z_mat = np.copy(z_matrix)
    n_row, n_column = x_mat.shape
    x_mat, z_mat, rank = slinalg.row_reduction(x_mat, z_mat)
    if x_mat[rank][n_column - 1] == 0:
        rank = rank - 1

    h_positions = _position_finder(x_mat)

    x_mat, z_mat = slinalg.hadamard_transform(x_mat, z_mat, h_positions)
    assert (np.linalg.det(x_mat)).astype(
        int
    ) % 2 != 0, "Stabilizer generators are not independent."
    x_inv = (np.linalg.det(x_mat.T) * np.linalg.inv(x_mat.T) % 2).astype(int)
    final_z = (z_mat.T @ x_inv) % 2

    # get position of non-zero diagonal elements in the final Z matrix to find qubits to apply clifford operations on
    # to remove those diagonal elements and make Z matrix a valid adjacency matrix
    final_z_diag = list(np.diag(final_z))
    z_diag_pos = [i for i, d in enumerate(final_z_diag) if d != 0]
    # remove diagonal parts of z_matrix
    for i in range(n_row):
        final_z[i, i] = 0

    state_graph = nx.from_numpy_array(final_z)

    assert np.array_equal(final_z, final_z.T), "final Z matrix in not a graph yet"
    x_mat = (x_inv @ x_mat.T) % 2
    assert (x_mat.shape[0] == x_mat.shape[1]) and (
        np.allclose(x_mat, np.eye(x_mat.shape[0]))
    ), "Unexpected X matrix."

    if get_ops_data:
        return state_graph, (h_positions, z_diag_pos)
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
    if isinstance(
        input_matrix, (np.ndarray, dmnp.ndarray)
    ):  # check if numpy array or numpy/jax array
        rho = input_matrix
    else:
        raise TypeError("Input density matrix must be a numpy.ndarray")
    n_qubits = int(np.round(np.log2(rho.shape[0])))

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
        final_state = dmf.apply_cz(final_state, mapping[edge[0]], mapping[edge[1]])

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

    :param input_matrix: a density matrix
    :type input_matrix: numpy.ndarray
    :return: a stabilizer representation
    :rtype: list[str]
    """
    # TODO: generalize to general stabilizer states
    graph = density_to_graph(input_matrix)
    stabilizer = graph_to_stabilizer(graph)
    return stabilizer


def _position_finder(x_matrix):
    """
    A helper function to obtain the position of the Hadamard gates needed to turn a stabilizer state into a graph state

    :param x_matrix: binary matrix for representing Pauli X part of the symplectic binary
            representation of the stabilizer generators
    :type x_matrix: numpy.ndarray
    :return: list of qubit positions to apply the hadamard on
    :rtype: list
    """
    pivot = [0, 0]
    n = x_matrix.shape[0]
    pos_list = []
    while pivot[1] < n and pivot[0] < n:
        try:
            if x_matrix[pivot[0] + 1, pivot[1]] == 1:
                pivot = [pivot[0] + 1, pivot[1]]
            if x_matrix[pivot[0] + 1, pivot[1] + 1] == 1:
                pivot = [pivot[0] + 1, pivot[1] + 1]
            else:
                pivot = [pivot[0], pivot[1] + 1]
                pos_list.append(pivot[1])
        except:
            break

    return pos_list


def state_to_graph(state):
    """
    A helper function to turn any valid representation into a graph. It also returns the StabilizerTableau corresponding
     to the initial state, and the gate sequence needed to convert the state into a graph-state.

    :param state: the state to be converted to graph
    :type state: StabilizerTableau or CliffordTableau or nx.Graph or np.ndarray
    :return: tuple (graph, input state's stabilizer tableau, gate_list)
    :rtype: tuple (nx.Graph, StabilizerTableau, list)
    """
    state = copy.deepcopy(state)
    # returns graph, input state's tableau, gate_list
    if isinstance(state, nx.Graph):
        tab = get_stabilizer_tableau_from_graph(state)
        return state, tab, []
    elif isinstance(state, np.ndarray):
        try:
            graph = nx.from_numpy_array(state)
            tab = get_stabilizer_tableau_from_graph(graph)
            return graph, [], tab
        except:
            raise ValueError(
                "the input numpy array is not a valid adjacency matrix, try fixing it or using other valid input types"
            )

    elif isinstance(state, CliffordTableau):
        z_matrix = state.stabilizer_z
        x_matrix = state.stabilizer_x
        tab = state.to_stabilizer()
    elif isinstance(state, StabilizerTableau):
        z_matrix = state.z_matrix
        x_matrix = state.x_matrix
        tab = state
    else:
        raise ValueError(
            "input data should either be a adjacency matrix, graph, Clifford or Stabilizer tableau"
        )
    graph, (h_pos, p_dag_pos) = _graph_finder(x_matrix, z_matrix, get_ops_data=True)
    gate_list = [("H", pos) for pos in h_pos] + [("P_dag", pos) for pos in p_dag_pos]

    # phase correction; adding Z gates at the end to make the phase of the transformed state equal to an ideal graph
    g_tab = get_stabilizer_tableau_from_graph(graph)
    phase_correction = _phase_correction(tab, g_tab, gate_list)

    gate_list += phase_correction
    return graph, tab, gate_list


def _phase_correction(stabilizer_tab1, stabilizer_tab2, gate_list):
    """
    If gate list transforms stabilizer generators of state 1 to state 2, then this function finds the list of gates
     needed to be added to the gate list to also have the phase of the 2 states exactly the same.
    The second state's stabilizer generators must represent a graph state.
     Returns a set of Z gates applied on appropriate qubits in the format of list of tuples [("Z", qubit_index)]

    :param stabilizer_tab1: stabilizer tableau for the initial state
    :type stabilizer_tab1: StabilizerTableau
    :param stabilizer_tab2: stabilizer tableau for the final state
    :type stabilizer_tab2: StabilizerTableau
    :param gate_list: a gate list, made up of gate tuples ("gate name", qubit index) that transforms initial state's
     stabilizer
    :type gate_list: list
    :return: a list of tuples [("Z", qubit_index)] to correct phase
    """

    tab1 = canonical_form(stabilizer_tab1)
    tab2 = canonical_form(stabilizer_tab2)
    new_tab = canonical_form(run_circuit(tab1.copy(), gate_list))
    phase_diff = (tab2.phase - new_tab.phase) % 2
    x_mat = np.copy(new_tab.x_matrix)
    x_inv = ((np.linalg.det(x_mat) * np.linalg.inv(x_mat)) % 2).astype(int)
    z_ops = (x_inv @ phase_diff) % 2
    phase_correction = [("Z", index) for index, z in enumerate(z_ops) if z]
    return phase_correction
