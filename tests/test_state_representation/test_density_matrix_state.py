from src.backends.density_matrix.functions import (
    is_psd,
    get_one_qubit_gate,
    get_two_qubit_controlled_gate,
    sigmax,
    sigmaz,
)
from src.backends.density_matrix.state import DensityMatrix
import numpy as np
import networkx as nx


def test_density_matrix_creation_ndarray():
    state_rep = DensityMatrix(np.eye(2))

    assert state_rep.data.shape[0] == state_rep.data.shape[1] == 2
    assert is_psd(state_rep.data)


def test_density_matrix_creation_graph():
    g = nx.Graph([(1, 2), (1, 3)])
    state_rep = DensityMatrix.from_graph(g)

    assert state_rep.data.shape[0] == state_rep.data.shape[1] == 8
    assert is_psd(state_rep.data)


def test_density_matrix_get_single_qubit_gate():
    # TODO: replace these prints by assert np.allclose( <>, expected)
    print(get_one_qubit_gate(3, 0, sigmaz()))
    print(get_two_qubit_controlled_gate(3, 0, 1, sigmax()))
