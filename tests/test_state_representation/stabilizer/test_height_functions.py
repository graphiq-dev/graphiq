import pytest
from graphiq.backends.stabilizer.functions.stabilizer import rref
from graphiq.backends.stabilizer.functions.height import height_func_list
from graphiq.backends.stabilizer.tableau import StabilizerTableau
import numpy as np
import networkx as nx


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
