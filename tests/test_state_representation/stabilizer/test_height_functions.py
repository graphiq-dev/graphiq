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
import networkx as nx
import numpy as np
import pytest

from graphiq.backends.stabilizer.functions.height import (
    height_dict,
    height_func_list,
    height_function,
)
from graphiq.backends.stabilizer.functions.stabilizer import rref
from graphiq.backends.stabilizer.tableau import StabilizerTableau


def case1():
    # a set of adjacency matrices (of graphs) for
    # which the height function is to be confirmed with Matlab code
    adj_matrix = np.array(
        [
            [0.0, 0.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0, 1.0, 0.0],
            [1.0, 1.0, 1.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 1.0, 0.0],
        ]
    ).astype(int)
    h_func = [0, 1, 2, 2, 1, 0]
    return adj_matrix, h_func


def case2():
    adj_matrix = np.array(
        [
            [0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0],
        ]
    ).astype(int)
    h_func = [0, 1, 1, 2, 3, 2, 1, 0]
    return adj_matrix, h_func


def case3():
    adj_matrix = np.array(
        [
            [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0],
            [1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        ]
    ).astype(int)
    h_func = [0, 1, 2, 3, 3, 3, 2, 2, 1, 0]
    return adj_matrix, h_func


def case4():
    adj_matrix = np.array(
        [
            [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
        ]
    ).astype(int)
    h_func = [0, 1, 2, 3, 4, 4, 4, 3, 2, 2, 1, 0]
    return adj_matrix, h_func


def case5():
    adj_matrix = np.array(
        [
            [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0],
            [0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0],
            [0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0],
            [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    ).astype(int)
    h_func = [0, 1, 2, 3, 4, 5, 4, 5, 4, 4, 3, 2, 1, 0]
    return adj_matrix, h_func


@pytest.mark.parametrize("case_number", [case1, case2, case3, case4, case5])
def test_height_comparison(case_number):
    adj_matrix, h_func = case_number()
    n_nodes = np.shape(adj_matrix)[0]
    h_x = [0] + height_func_list(np.eye(n_nodes), adj_matrix)
    assert h_x == h_func


def test_paper_example():
    # This corresponds to the echelon gauge transformation from Fig.1(b) to Fig.1(c)(i)
    # of Li et al.'s supplementary material.

    # check matrices of Fig.1(b)
    z_matrix_in = np.array(
        [
            [0, 1, 1, 0],
            [1, 0, 1, 1],
            [1, 1, 0, 1],
            [0, 1, 1, 0],
        ]
    ).astype(int)
    x_matrix_in = np.eye(4, 4)
    r_vector_in = np.zeros(4)
    tableau = StabilizerTableau([x_matrix_in, z_matrix_in], r_vector_in)
    # Get echelon gauge matrices
    tableau = rref(tableau)

    # Height function of before and after gauge transformation.
    # Should be the same as height function also does the transformation.
    height_list_1 = height_func_list(x_matrix_in, z_matrix_in)
    height_list_1 = np.array(height_list_1)
    height_list_2 = height_func_list(tableau.x_matrix, tableau.z_matrix)
    height_list_2 = np.array(height_list_2)

    # Check matrices of Fig.1(c)(i) which corresponds to after echelon gauge transformation
    z_matrix_expected = np.array(
        [
            [0, 1, 1, 0],
            [1, 0, 1, 1],
            [0, 1, 1, 0],
            [0, 1, 1, 0],
        ]
    ).astype(int)

    x_matrix_expected = np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 1],
        ]
    ).astype(int)
    # Corresponding height function, ignoring site 0
    height_list_expected = np.array([1, 2, 1, 0])

    assert (height_list_1 == height_list_expected).all()
    assert (height_list_2 == height_list_expected).all()
    assert (tableau.x_matrix == x_matrix_expected).all()
    assert (tableau.z_matrix == z_matrix_expected).all()


@pytest.mark.parametrize("qubit_position, expected", [(0, 1), (1, 1), (2, 0)])
def test_height_function_linear_cluster_3qubit(qubit_position, expected):
    # the 3-qubit linear cluster state: x is the identity, z is path_graph(3)'s adjacency matrix
    x_matrix = np.eye(3)
    z_matrix = nx.to_numpy_array(nx.path_graph(3))
    assert height_function(x_matrix, z_matrix, qubit_position) == expected


def test_height_dict_graph_arg_must_be_networkx_graph():
    # a truthy value that is not a networkx graph is rejected explicitly
    with pytest.raises(ValueError, match="valid networkx graph"):
        height_dict(graph="not a graph")


def test_height_dict_needs_graph_or_matrices():
    # an empty (falsy) graph argument falls through to the "neither" branch
    with pytest.raises(ValueError, match="Either a graph or both"):
        height_dict(graph=[])


def test_height_dict_from_graph_orders_nodes_by_label():
    # Nodes are added in the order 0, 2, 1, 3; height_dict(graph=...) must key its result by
    # node label, not by the order the edges were inserted in.
    graph = nx.Graph()
    graph.add_edges_from([(0, 2), (1, 3)])

    h_dict = height_dict(graph=graph)

    assert [h_dict[0], h_dict[1], h_dict[2], h_dict[3]] == [1, 2, 1, 0]
