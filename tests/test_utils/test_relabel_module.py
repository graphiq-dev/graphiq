import pytest

from src.utils.relabel_module import *
import networkx as nx
import numpy as np


def random_connected_graph(n, p, mode=1, seed=None):
    rnd_maker = nx.gnp_random_graph
    G = nx.Graph()
    G.add_edges_from(((0, 1), (2, 3)))
    while not nx.is_connected(G):
        G = rnd_maker(n, p, seed=seed, directed=False)
    adj_G = nx.to_numpy_array(G)
    return adj_G


@pytest.mark.parametrize(
    "n_node",
    [
        5,
        6,
        7,
        8,
    ],
)
def test_random_graph(n_node):
    n_iso = np.math.factorial(n_node - 2)
    for p in range(4, 7):
        found = []
        for i in range(3):
            adj = random_connected_graph(n_node, p / 10)
            iso_found = len(iso_finder(adj, n_iso)) / n_iso
            found.append(iso_found)
        avg = (sum(found) / len(found))
        assert 0 <= avg <= 1

@pytest.mark.parametrize(
    "n_node",
    [
        11,
        12,
        13,
        15,
    ],
)
def test_relabel(n_node):
    # find 1000 relabeled graphs for an initial random one with n_nodes vertices.
    adj = random_connected_graph(n_node, 0.5)
    n_iso = 1000
    adj_arr = iso_finder(adj, n_iso)
    assert 1 <= len(adj_arr) <= 1000
    assert adj_arr[10].shape[1] == n_node


def test_complete_graph():
    # there should be no valid result for relabeling of a complete graph
    for n_node in range(5, 15):
        g = nx.complete_graph(n_node)
        adj = nx.to_numpy_array(g)
        n_iso = 10
        adj_arr = iso_finder(adj, n_iso)
        assert len(adj_arr) == 0


