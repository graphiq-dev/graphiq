import pytest
import numpy as np
import networkx as nx

from src.state import QuantumState


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
def test_initializing_dm_1(n):
    data = np.eye(2**n)
    state = QuantumState(n, data, representation="density matrix")
    assert np.allclose(state.dm.data, data / np.trace(data))


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
def test_initializing_dm_2(n):
    data = np.eye(2**n)
    state = QuantumState(n, [data], representation=["density matrix"])
    assert np.allclose(state.dm.data, data / np.trace(data))


def test_initializing_dm_3():
    """Test that it fails if we don't provide the right datatype"""
    data = "bad input"
    with pytest.raises(TypeError):
        QuantumState(2, data, representation="density matrix")


def test_initializing_graph_1():
    n = 1
    data = 5
    state = QuantumState(n, data, representation="graph")
    assert len(state.graph.data.nodes) == n


def test_initializing_graph_2():
    n = 3
    data = nx.Graph([(1, 2), (2, 3)])
    state = QuantumState(n, data, representation="graph")
    assert len(state.graph.data.nodes) == n


def test_initializing_dm_and_graph():
    n = 3
    data_g = nx.Graph([(1, 2), (2, 3)])
    data_dm = np.eye(2**n)
    state = QuantumState(
        n, [data_dm, data_g], representation=["density matrix", "graph"]
    )
    assert len(state.graph.data.nodes) == n
    assert np.allclose(state.dm.data, data_dm / np.trace(data_dm))
