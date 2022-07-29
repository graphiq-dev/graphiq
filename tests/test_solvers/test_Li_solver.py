import pytest
from src.backends.stabilizer.functions.matrix_functions import rref
from src.backends.stabilizer.functions.height_function import height_func_list
import numpy as np
import networkx as nx

# a set of adjacency matrices (of graphs) for which the height function is to be confirmed with Matlab code
adj_matrix_1 = np.array(
    [
        [0.0, 0.0, 1.0, 1.0, 1.0],
        [0.0, 0.0, 1.0, 1.0, 0.0],
        [1.0, 1.0, 0.0, 1.0, 0.0],
        [1.0, 1.0, 1.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, 1.0, 0.0],
    ]
)
adj_matrix_2 = np.array(
    [
        [0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0],
        [1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0],
        [1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0],
    ]
)
adj_matrix_3 = np.array(
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
)
adj_matrix_4 = np.array(
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
)
adj_matrix_5 = np.array(
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
)
h_1 = [0, 1, 2, 2, 1, 0]
h_2 = [0, 1, 1, 2, 3, 2, 1, 0]
h_3 = [0, 1, 2, 3, 3, 3, 2, 2, 1, 0]
h_4 = [0, 1, 2, 3, 4, 4, 4, 3, 2, 2, 1, 0]
h_5 = [0, 1, 2, 3, 4, 5, 4, 5, 4, 4, 3, 2, 1, 0]


@pytest.mark.parametrize("case_number", [1, 2, 3, 4, 5])
def test_height_comparison(case_number):
    n_nodes = np.shape(globals()[f"adj_matrix_{case_number}"])[0]
    h_x = [0] + height_func_list(
        np.eye(n_nodes), globals()[f"adj_matrix_{case_number}"]
    )
    assert h_x == globals()[f"h_{case_number}"]


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
    print("1", "\n", x_1.astype(int), "\n", z_1)
    for i in range(n_nodes):
        pivot[1] = i
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
        print(f"pivot ={pivot}")
        assert pivot[0] - old_pivot <= 2
        for j in range(1 + int(pivot[0]), n_nodes):
            assert int(x_1[j, i]) == 0 and int(z_1[j, i]) == 0
    pivot = [0, 0]
    print("2", "\n", x_2.astype(int), "\n", z_2)
    for i in range(n_nodes):
        pivot[1] = i
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
        print(f"pivot ={pivot}")
        assert pivot[0] - old_pivot <= 2
        for j in range(int(pivot[0]) + 1, n_nodes):
            assert int(x_2[j, i]) == 0 and int(z_2[j, i]) == 0


def test_Li_et_al_example():
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
    )
    x_matrix_in = np.eye(4, 4)
    r_vector_in = np.zeros([4, 1])

    # Get echelon gauge matrices
    x_matrix, z_matrix, _ = rref(x_matrix_in, z_matrix_in, r_vector_in)

    # Height function of before and after gauge transformation. Should be the same as
    # height function also does the transformation.
    height_list_1 = height_func_list(x_matrix_in, z_matrix_in)
    height_list_1 = np.array(height_list_1)
    height_list_2 = height_func_list(x_matrix, z_matrix)
    height_list_2 = np.array(height_list_2)

    # Check matrices of Fig.1(c)(i) which corresponds to after echelon gauge transformation
    z_matrix_Li = np.array(
        [
            [0, 1, 1, 0],
            [1, 0, 1, 1],
            [0, 1, 1, 0],
            [0, 1, 1, 0],
        ]
    )
    x_matrix_Li = np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 1],
        ]
    )
    # Corresponding height function, ignoring site 0
    height_list_Li = np.array([1, 2, 1, 0])

    assert (height_list_1 == height_list_Li).all()
    assert (height_list_2 == height_list_Li).all()
    assert (x_matrix == x_matrix_Li).all()
    assert (z_matrix == z_matrix_Li).all()
