"""
Some conversion functions between different tableau types or between tableau and graph
"""
import networkx as nx
import numpy as np

import src.backends.stabilizer.functions.clifford as sfc
from src.backends.stabilizer.functions.stabilizer import inverse_circuit
from src.backends.stabilizer.tableau import StabilizerTableau


def clifford_from_stabilizer(stabilizer_tableau):
    """
    Return the CliffordTableau from the given StabilizerTableau after finding destabilizers

    :param stabilizer_tableau: a stabilizer Tableau
    :type stabilizer_tableau: StabilizerTableau
    :return: the Clifford
    :rtype: CliffordTableau
    """
    n_qubits = stabilizer_tableau.n_qubits
    _, circuit = inverse_circuit(stabilizer_tableau)
    circuit.reverse()
    clifford_tableau = sfc.create_n_ket0_state(n_qubits)
    return sfc.run_circuit(clifford_tableau, circuit)


def get_stabilizer_tableau_from_graph(graph):
    """
    Create a stabilizer tableau from a networkx graph

    :param graph: a network graph
    :type graph: networkx.Graph
    :return: a stabilizer tableau representing the graph state
    :rtype: StabilizerTableau
    """
    adj_matrix = nx.to_numpy_array(graph)
    n_qubits = graph.number_of_nodes()
    tableau = StabilizerTableau(n_qubits)
    tableau.x_matrix = np.eye(n_qubits).astype(int)
    tableau.z_matrix = adj_matrix.astype(int)
    return tableau


def get_clifford_tableau_from_graph(graph):
    """
    Create a Clifford tableau from a networkx graph

    :param graph: a network graph
    :type graph: networkx.Graph
    :return: a Clifford tableau representing the graph state
    :rtype: CliffordTableau
    """

    tableau = get_stabilizer_tableau_from_graph(graph)
    return clifford_from_stabilizer(tableau)
