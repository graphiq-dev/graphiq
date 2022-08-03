import pytest
import numpy as np
import networkx as nx
import src.backends.stabilizer.functions.conversion as sfc
import src.backends.stabilizer.functions.validation as sfv

import src.backends.stabilizer.functions.transformations as sft
from src.backends.stabilizer.functions.matrix_functions import rref
from src.backends.stabilizer.functions.height_function import height_func_list
from functools import reduce


def test_symplectic_to_string():
    x_matrix1 = np.zeros((4, 4))
    z_matrix1 = np.eye(4)
    expected1 = ["ZIII", "IZII", "IIZI", "IIIZ"]
    assert reduce(
        lambda x, y: x and y,
        map(
            lambda a, b: a == b,
            sfc.symplectic_to_string(x_matrix1, z_matrix1),
            expected1,
        ),
        True,
    )

    x_matrix2 = np.array([[1, 0, 0, 0], [0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
    z_matrix2 = np.eye(4)
    expected2 = ["YIII", "IZII", "IXZI", "IIXZ"]
    assert reduce(
        lambda x, y: x and y,
        map(
            lambda a, b: a == b,
            sfc.symplectic_to_string(x_matrix2, z_matrix2),
            expected2,
        ),
        True,
    )


def test_string_to_symplectic():
    generator_list = ["ZIII", "IZII", "IIZI", "IIIZ"]
    x_matrix, z_matrix = sfc.string_to_symplectic(generator_list)
    assert np.array_equal(x_matrix, np.zeros((4, 4)))
    assert np.array_equal(z_matrix, np.eye(4))


def test_symplectic_product():
    adj_matrix = np.array(
        [[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]]
    ).astype(int)
    tableau = np.block([[np.eye(4), adj_matrix]])
    dim = 2

    assert np.array_equal(
        sfv.binary_symplectic_product(tableau, tableau), np.zeros((4, 4))
    )
    assert sfv.is_symplectic_self_orthogonal(tableau)


@pytest.mark.parametrize("n_nodes", [*range(5, 15)])
def test_echelon_form(n_nodes):
    g1 = nx.complete_graph(n_nodes)
    g2 = nx.gnp_random_graph(n_nodes, 0.5)
    r_vector_1 = np.zeros([n_nodes, 1])
    r_vector_2 = np.zeros([n_nodes, 1])
    z_1 = nx.to_numpy_array(g1).astype(int)
    z_2 = nx.to_numpy_array(g2).astype(int)
    x_1 = np.eye(n_nodes)
    x_2 = np.eye(n_nodes)

    x_1, z_1, r_vector_1 = rref(x_1, z_1, r_vector_1)
    x_2, z_2, r_vector_2 = rref(x_2, z_2, r_vector_2)
    pivot = [0, 0]
    # print("1", "\n", x_1.astype(int), "\n", z_1)
    for i in range(n_nodes):
        # pivot[1] = i
        old_pivot = pivot[0]
        nonzero_x1 = np.nonzero(x_1[:, i])[0]
        nonzero_z1 = np.nonzero(z_1[:, i])[0]
        if len(nonzero_x1) == 0 or len(nonzero_z1) == 0:
            if len(nonzero_x1) == 0 and len(nonzero_z1) == 0:
                pivot[0] = old_pivot
            elif len(nonzero_z1) == 0:
                pivot[0] = max(nonzero_x1[-1], old_pivot)
            else:
                pivot[0] = max(nonzero_z1[-1], old_pivot)
        else:
            pivot[0] = max(nonzero_x1[-1], nonzero_z1[-1], old_pivot)
        # print(f"pivot ={pivot}")
        assert pivot[0] - old_pivot <= 2
        for j in range(1 + int(pivot[0]), n_nodes):
            assert int(x_1[j, i]) == 0 and int(z_1[j, i]) == 0
    pivot = [0, 0]
    # print("2", "\n", x_2.astype(int), "\n", z_2)
    for i in range(n_nodes):
        # pivot[1] = i
        old_pivot = pivot[0]
        nonzero_x2 = np.nonzero(x_2[:, i])[0]
        nonzero_z2 = np.nonzero(z_2[:, i])[0]
        if len(nonzero_x2) == 0 or len(nonzero_z2) == 0:
            if len(nonzero_x2) == 0 and len(nonzero_z2) == 0:
                pivot[0] = old_pivot
            elif len(nonzero_z2) == 0:
                pivot[0] = max(nonzero_x2[-1], old_pivot)
            else:
                pivot[0] = max(nonzero_z2[-1], old_pivot)
        else:
            pivot[0] = max(nonzero_x2[-1], nonzero_z2[-1], old_pivot)
        # print(f"pivot ={pivot}")
        assert pivot[0] - old_pivot <= 2
        for j in range(int(pivot[0]) + 1, n_nodes):
            assert int(x_2[j, i]) == 0 and int(z_2[j, i]) == 0


def test_qubit_insertion():
    tableau = sft.create_n_plus_state(4)
    tableau = sft.insert_qubit(tableau, 1)


def test_qubit_insertion2():
    tableau = sft.create_n_ket0_state(4)
    tableau = sft.insert_qubit(tableau, 1)
    assert np.array_equal(tableau.table, sft.create_n_ket0_state(5).table)


def test_qubit_removal():
    tableau = sft.create_n_plus_state(4)
    tableau = sft.remove_qubit(tableau, 1)


def test_qubit_insertion_removal():
    tableau = sft.create_n_plus_state(4)
    tableau = sft.insert_qubit(tableau, 1)
    sft.remove_qubit(tableau, 1)
