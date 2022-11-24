import pytest
import numpy as np
import networkx as nx

from src.backends.density_matrix import numpy as dmnp
from src.state import QuantumState
from src.state import DENSITY_MATRIX_QUBIT_THRESH
from src.backends.stabilizer.tableau import CliffordTableau


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
def test_initializing_dm_1(n):
    data = dmnp.eye(2**n)
    state = QuantumState(n, data, representation="density matrix")
    assert np.allclose(state.dm.data, data / np.trace(data))


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
def test_initializing_dm_2(n):
    data = dmnp.eye(2**n)
    state = QuantumState(n, [data], representation=["density matrix"])
    assert np.allclose(state.dm.data, data / np.trace(data))


def test_initializing_dm_3():
    """Test that it fails if we don't provide the right datatype"""
    data = "bad input"
    with pytest.raises(TypeError):
        QuantumState(2, data, representation="density matrix")


def test_initializing_dm_4():
    """Test that it fails if we have a size mismatch between state n and the data"""
    n = 5
    data = dmnp.eye(2 ** (n + 1))
    with pytest.raises(AssertionError):
        QuantumState(n, data, representation="density matrix")


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


def test_initializing_graph_3():
    """Test that it fails if we have a size mismatch between state n and the data"""
    n = 5
    data = nx.star_graph(n + 1)
    with pytest.raises(AssertionError):
        QuantumState(n, data, representation="graph")


def test_initializing_dm_and_graph():
    n = 3
    data_g = nx.Graph([(1, 2), (2, 3)])
    data_dm = dmnp.eye(2**n)
    state = QuantumState(
        n, [data_dm, data_g], representation=["density matrix", "graph"]
    )
    assert len(state.graph.data.nodes) == n
    assert np.allclose(state.dm.data, data_dm / np.trace(data_dm))


def test_initializing_stabilizer_1():
    """Checks that we can initialize stabilizers with integers"""
    n = 130
    state = QuantumState(n, n, representation="stabilizer")
    assert state.stabilizer.n_qubit == n


def test_initializing_stabilizer_2():
    """Checks that we can initialize stabilizers with a CliffordTableau"""
    n = 130
    state = QuantumState(n, CliffordTableau(n), representation="stabilizer")
    assert state.stabilizer.n_qubit == n


def test_initializing_stabilizer_3():
    """Test that it fails if we have a size mismatch between state n and the data"""
    n = 5
    data = CliffordTableau(n + 1)
    with pytest.raises(AssertionError):
        QuantumState(n, data, representation="stabilizer")


@pytest.mark.parametrize("size, data", [(5, 5), (4, dmnp.eye(2**4))])
def test_automatic_representation_selection_1(size, data):
    """
    Test that it defaults to density matrix if the datatype is compatible and the state size is smaller than the
    threshold
    """
    state = QuantumState(size, data)
    assert state._dm is not None
    assert state._stabilizer is None
    assert state._graph is None


@pytest.mark.parametrize(
    "size, data", [(DENSITY_MATRIX_QUBIT_THRESH, np.array([])), (131, np.array([]))]
)
def test_automatic_representation_selection_2(size, data):
    """
    Tests that the QuantumState initialization will fail because the states being initialized are too large for density matrix representation,
    and the datatype passed is only compatible with density matrix representation

    Note: the data doesn't actually match the dimension specified to QuantumState, for computational efficiency. This should not matter
    to the test
    """
    with pytest.raises(ValueError):
        QuantumState(size, data)


@pytest.mark.parametrize(
    "size, data",
    [
        (DENSITY_MATRIX_QUBIT_THRESH, DENSITY_MATRIX_QUBIT_THRESH),
        (33, 33),
        (1092, 1092),
    ],
)
def test_automatic_representation_selection_3(size, data):
    """Tests that large states with input data that matches all representations will be initialized in Stabilizer Formalism"""
    state = QuantumState(size, data)
    assert state._dm is None
    assert state._stabilizer is not None
    assert state._graph is None


@pytest.mark.parametrize(
    "size, data",
    [
        (DENSITY_MATRIX_QUBIT_THRESH, nx.star_graph(DENSITY_MATRIX_QUBIT_THRESH - 1)),
        (33, nx.star_graph(33 - 1)),
        (1092, nx.star_graph(1092 - 1)),
    ],
)
def test_automatic_representation_selection_3(size, data):
    """Tests that large states with input data that matches graph representation will be initialized in graph rep"""
    state = QuantumState(size, data)
    assert state._dm is None
    assert state._stabilizer is None
    assert state._graph is not None
