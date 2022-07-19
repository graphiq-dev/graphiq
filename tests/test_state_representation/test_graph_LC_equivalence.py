import pytest
from src.backends.lc_equivalence_check import *
import matplotlib.pyplot as plt
import numpy as np


def _tester(n):
    """
    A test function for internal purpose. Searches over all random graphs of size "n" to finally find two that are
        LC equivalent. Should not be used for n > 7 since it may never find a solution in reasonable time.

    :param n: the size of the graphs (number of the nodes/ qubits)
    :type n: int
    :return: solution, G1, G2. Two graphs that are LC equivalent and the Clifford operator needed for the transformation
    :rtype: numpy.ndarray, networkx.Graph, networkx.Graph
    """
    solution = None
    while not isinstance(solution, np.ndarray):
        g1, g2 = nx.fast_gnp_random_graph(n, 0.65), nx.fast_gnp_random_graph(n, 0.65)

        z_1 = nx.to_numpy_array(g1).astype(int)
        z_2 = nx.to_numpy_array(g2).astype(int)

        success, solution = is_lc_equivalent(z_1, z_2, mode="deterministic")

        if isinstance(solution, np.ndarray):
            plt.figure(1)
            nx.draw(g1, with_labels=True)
            plt.figure(2)
            nx.draw(g2, with_labels=True)
            print(local_clifford_ops(solution), "\n")

    return solution, (g1, g2)


@pytest.mark.parametrize("n_nodes", [2, 3, 4])
def test_equivalence(n_nodes):
    solution, (graph1, graph2) = _tester(n_nodes)
    z_1 = nx.to_numpy_array(graph1).astype(int)

    g_list = lc_graph_operations(z_1, solution)
    print("LC operations needed on nodes:", g_list)
    g_new = graph1
    for i in g_list:
        g_new = local_comp_graph(g_new, i)

    z_new = nx.to_numpy_array(g_new)
    z_2 = nx.to_numpy_array(graph2).astype(int)
    assert nx.is_isomorphic(g_new, graph2), "found graphs are not LC equivalent"
    assert np.array_equal(z_new, z_2), "found graphs are not LC equivalent"


@pytest.mark.parametrize("n_nodes", [2, 3, 4, 5, 6])
def test_star_graph(n_nodes):
    g1 = nx.complete_graph(n_nodes)
    g2 = nx.star_graph(n_nodes - 1)
    z_1 = nx.to_numpy_array(g1).astype(int)
    z_2 = nx.to_numpy_array(g2).astype(int)
    success, solution = is_lc_equivalent(z_1, z_2, mode="deterministic")
    if isinstance(solution, np.ndarray):
        print(local_clifford_ops(solution), "\n")
    assert isinstance(solution, np.ndarray) or solution is None
    is_equivalent, _ = iso_equal_check(g1, g2)
    assert is_equivalent
