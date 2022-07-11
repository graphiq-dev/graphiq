import pytest
import networkx as nx

from src.backends.graph.state import Graph


@pytest.fixture(scope="function")
def graph_rep_1():
    g = nx.Graph([(1, 2), (2, 3)])
    return Graph(data=g, root_node_id=1)


@pytest.fixture(scope="function")
def graph_rep_2():
    g = nx.Graph(
        [(frozenset([1, 2, 3]), frozenset([4, 5, 6])), (frozenset([1, 2, 3]), 7)]
    )
    return Graph(g, root_node_id=7)
