import networkx as nx

import graphiq.backends.density_matrix.functions as dmf
from graphiq.backends.density_matrix import numpy as np
from graphiq.backends.density_matrix.state import DensityMatrix


# @pytest.fixture(params=['numpy_fixture', 'jax_fixture'])
# def library_fixture(request):
#     return request.getfixturevalue(request.param)
# import graphiq.backends.density_matrix
# from graphiq.backends.density_matrix import numpy as np
# import graphiq.backends.density_matrix.functions as dmf


def test_density_matrix_creation_ndararay():
    state_rep = DensityMatrix(np.eye(2))

    assert state_rep.data.shape[0] == state_rep.data.shape[1] == 2
    assert dmf.is_psd(state_rep.data)


def test_density_matrix_creation_graph():
    g = nx.Graph([(1, 2), (1, 3)])
    state_rep = DensityMatrix.from_graph(g)

    assert state_rep.data.shape[0] == state_rep.data.shape[1] == 8
    assert dmf.is_psd(state_rep.data)


def test_density_matrix_get_single_qubit_gate():
    # TODO: replace these prints by assert np.allclose( <>, expected)
    print(dmf.get_one_qubit_gate(3, 0, dmf.sigmaz()))
    print(dmf.get_two_qubit_controlled_gate(3, 0, 1, dmf.sigmax()))
