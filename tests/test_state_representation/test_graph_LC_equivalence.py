from src.backends.graph.LC_equivalence import *
from src.backends.graph.LC_equivalence import _tester
import numpy as np


def test_equivalence():
    for n in range(2, 7):
        solution, (G1, G2) = _tester(n)
        z_1 = nx.to_numpy_array(G1).astype(int)
        z_2 = nx.to_numpy_array(G2).astype(int)

        g_list = LC_graph_operations(z_1, solution)
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
        solution = solver(z_1, z_2, Mode='deterministic')
        if isinstance(solution, type(np.array([0]))):
            print(local_clifford_ops(solution), "\n")
        assert isinstance(solution, type(np.array([0]))) or isinstance(solution, type(None))
        a, _ = iso_equal_check(g1, g2)
        assert a