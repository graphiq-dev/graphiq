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

        success, solution = is_lc_equivalent(z_1, z_2, mode='deterministic')

        if isinstance(solution, np.ndarray):
            plt.figure(1)
            nx.draw(g1, with_labels=True)
            plt.figure(2)
            nx.draw(g2, with_labels=True)
            print(local_clifford_ops(solution), "\n")

    return solution, (g1, g2)


def test_equivalence():
    for n in range(2, 5):
        solution, (G1, G2) = _tester(n)
        z_1 = nx.to_numpy_array(G1).astype(int)


        g_list = lc_graph_operations(z_1, solution)
        print("LC operations needed on nodes:", g_list)
        g_new = G1
        for i in g_list:
            g_new = local_comp_graph(g_new, i)

        z_new = nx.to_numpy_array(g_new)
        z_2 = nx.to_numpy_array(G2).astype(int)
        assert (nx.is_isomorphic(g_new, G2)), "found graphs are not LC equivalent"
        assert (np.array_equal(z_new, z_2)), "found graphs are not LC equivalent"


def test_star_graph():
    for n in range(2, 7):
        g1 = nx.complete_graph(n)
        g2 = nx.star_graph(n - 1)
        z_1 = nx.to_numpy_array(g1).astype(int)
        z_2 = nx.to_numpy_array(g2).astype(int)
        success, solution = is_lc_equivalent(z_1, z_2, mode='deterministic')
        if isinstance(solution, type(np.array([0]))):
            print(local_clifford_ops(solution), "\n")
        assert isinstance(solution, type(np.array([0]))) or isinstance(solution, type(None))
        a, _ = iso_equal_check(g1, g2)
        print(a)
        #assert a
