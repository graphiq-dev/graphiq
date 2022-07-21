import pytest
from src.backends.stabilizer.echelon_gauge import height_func_list
from src.backends.stabilizer.echelon_gauge import rref
import numpy as np
import networkx as nx

# a set of adjacency matrices (of graphs) for which the height function is to be confirmed with Matlab code
adj_matrix_1 = np.array([[0., 0., 1., 1., 1.],
                         [0., 0., 1., 1., 0.],
                         [1., 1., 0., 1., 0.],
                         [1., 1., 1., 0., 1.],
                         [1., 0., 0., 1., 0.]])
adj_matrix_2 = np.array([[0., 1., 1., 0., 1., 1., 0.],
                         [1., 0., 1., 0., 1., 1., 0.],
                         [1., 1., 0., 0., 1., 0., 1.],
                         [0., 0., 0., 0., 0., 0., 1.],
                         [1., 1., 1., 0., 0., 1., 0.],
                         [1., 1., 0., 0., 1., 0., 1.],
                         [0., 0., 1., 1., 0., 1., 0.]])
adj_matrix_3 = np.array([[0., 1., 0., 1., 0., 1., 0., 1., 1.],
                         [1., 0., 0., 1., 1., 0., 0., 0., 1.],
                         [0., 0., 0., 1., 0., 1., 0., 0., 1.],
                         [1., 1., 1., 0., 1., 0., 0., 0., 1.],
                         [0., 1., 0., 1., 0., 0., 0., 0., 0.],
                         [1., 0., 1., 0., 0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0., 0., 0., 0., 1.],
                         [1., 0., 0., 0., 0., 0., 0., 0., 0.],
                         [1., 1., 1., 1., 0., 0., 1., 0., 0.]])
adj_matrix_4 = np.array([[0., 1., 1., 1., 1., 1., 0., 1., 1., 1., 0.],
                         [1., 0., 0., 0., 1., 1., 0., 0., 1., 1., 1.],
                         [1., 0., 0., 0., 1., 0., 1., 1., 1., 1., 1.],
                         [1., 0., 0., 0., 0., 0., 1., 1., 1., 1., 0.],
                         [1., 1., 1., 0., 0., 0., 0., 1., 0., 0., 0.],
                         [1., 1., 0., 0., 0., 0., 1., 0., 1., 1., 0.],
                         [0., 0., 1., 1., 0., 1., 0., 1., 1., 1., 1.],
                         [1., 0., 1., 1., 1., 0., 1., 0., 0., 0., 1.],
                         [1., 1., 1., 1., 0., 1., 1., 0., 0., 0., 0.],
                         [1., 1., 1., 1., 0., 1., 1., 0., 0., 0., 0.],
                         [0., 1., 1., 0., 0., 0., 1., 1., 0., 0., 0.]])
adj_matrix_5 = np.array([[0., 0., 1., 0., 0., 1., 1., 0., 1., 0., 1., 0., 1.],
                         [0., 0., 1., 0., 1., 0., 0., 1., 1., 0., 1., 1., 0.],
                         [1., 1., 0., 0., 1., 0., 0., 0., 1., 1., 1., 1., 0.],
                         [0., 0., 0., 0., 1., 0., 1., 1., 0., 0., 0., 1., 1.],
                         [0., 1., 1., 1., 0., 1., 0., 1., 0., 0., 0., 1., 0.],
                         [1., 0., 0., 0., 1., 0., 0., 1., 0., 1., 0., 0., 0.],
                         [1., 0., 0., 1., 0., 0., 0., 0., 1., 0., 1., 1., 1.],
                         [0., 1., 0., 1., 1., 1., 0., 0., 1., 1., 1., 1., 1.],
                         [1., 1., 1., 0., 0., 0., 1., 1., 0., 1., 1., 1., 0.],
                         [0., 0., 1., 0., 0., 1., 0., 1., 1., 0., 0., 1., 0.],
                         [1., 1., 1., 0., 0., 0., 1., 1., 1., 0., 0., 0., 0.],
                         [0., 1., 1., 1., 1., 0., 1., 1., 1., 1., 0., 0., 0.],
                         [1., 0., 0., 1., 0., 0., 1., 1., 0., 0., 0., 0., 0.]])
h_1 = [0, 1, 2, 2, 1, 0]
h_2 = [0, 1, 1, 2, 3, 2, 1, 0]
h_3 = [0, 1, 2, 3, 3, 3, 2, 2, 1, 0]
h_4 = [0, 1, 2, 3, 4, 4, 4, 3, 2, 2, 1, 0]
h_5 = [0, 1, 2, 3, 4, 5, 4, 5, 4, 4, 3, 2, 1, 0]


@pytest.mark.parametrize("case_number", [1, 2, 3, 4, 5])
def test_height_comparison(case_number):
    number_of_nodes = np.shape(globals()[f'adj_matrix_{case_number}'])[0]
    h_x = [0] + height_func_list(np.eye(number_of_nodes), globals()[f'adj_matrix_{case_number}'])
    assert h_x == globals()[f'h_{case_number}']


@pytest.mark.parametrize("n_nodes", [6, 7, 8, 9, 10])
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
    print('\n', x_1, '\n',z_1)
    for i in range(n_nodes):
        pivot[1] = i
        old_pivot = pivot[0]
        pivot[0] = max(np.nonzero(x_1[:, i])[0][-1], np.nonzero(z_1[:, i])[0][-1], old_pivot)
        assert pivot[0] - old_pivot <= 2
        for j in range(1 + int(pivot[0]), n_nodes):
            assert x_1[j, i] == 0 and z_1[j, i] == 0
    pivot = [0, 0]
    for i in range(n_nodes):
        pivot[1] = i
        old_pivot = pivot[0]
        pivot[0] = max(np.nonzero(x_2[:, i])[0][-1], np.nonzero(z_2[:, i])[0][-1], old_pivot)
        assert pivot[0] - old_pivot <= 2
        for j in range(pivot[0] + 1, n_nodes):
            assert x_1[j, i] == 0 and z_1[j, i] == 0
