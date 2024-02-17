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
"""
Some rc functions between different tableau types or between tableau and graph
"""

import networkx as nx
import numpy as np

import graphiq.backends.stabilizer.functions.clifford as sfc
import graphiq.backends.stabilizer.functions.transformation as transform
from graphiq.backends.stabilizer.functions.stabilizer import inverse_circuit
from graphiq.backends.stabilizer.tableau import StabilizerTableau


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
    clifford_tableau = sfc.create_n_ket0_state(n_qubits)
    return transform.run_circuit(clifford_tableau, circuit, reverse=True)


def stabilizer_from_clifford(clifford_tableau):
    """
    Return the StabilizerTableau from the given CliffordTableau

    :param clifford_tableau: a clifford Tableau
    :type clifford_tableau: CliffordTableau
    :return: the Stabilizer
    :rtype: StabilizerTableau
    """
    return clifford_tableau.to_stabilizer()


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
