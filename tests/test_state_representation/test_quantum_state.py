import pytest
import numpy as np
import networkx as nx

from src.backends.density_matrix import numpy as dmnp
from src.state import QuantumState
from src.state import DENSITY_MATRIX_QUBIT_THRESH
from src.backends.stabilizer.clifford_tableau import CliffordTableau


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
def test_initializing_dm_1(n):
    data = dmnp.eye(2**n)
    state = QuantumState(data, rep_type="dm")
    assert np.allclose(state.rep_data.data, data / np.trace(data))


def test_initializing_dm_2():
    """Test that it fails if we don't provide the right datatype"""
    data = "bad input"
    with pytest.raises(TypeError):
        QuantumState(data, rep_type="dm")


def test_initializing_graph_1():
    data = 5
    state = QuantumState(data, rep_type="g")
    assert len(state.rep_data.data.nodes) == data


def test_initializing_graph_2():
    n = 3
    data = nx.Graph([(1, 2), (2, 3)])
    state = QuantumState(data, rep_type="g")
    assert len(state.rep_data.data.nodes) == n


def test_initializing_stabilizer_1():
    """Checks that we can initialize stabilizers with integers"""
    n = 130
    state = QuantumState(n, rep_type="s")
    assert state.rep_data.n_qubits == n


def test_initializing_stabilizer_2():
    """Checks that we can initialize stabilizers with a CliffordTableau"""
    n = 130
    state = QuantumState(CliffordTableau(n), rep_type="s")
    assert state.rep_data.n_qubits == n
