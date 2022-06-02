import numpy as np
import networkx as nx

from src.backends.density_matrix.state import DensityMatrix
from src.backends.graph.state import Graph
import src.backends.density_matrix.functions as dmf
import src.backends.graph.functions as gf
import src.backends.stabilizer.functions as sf
import qutip as qt
import src.backends.state_representation_conversion as sconverter


def test_negativity():
    n_qubits = 4

    st1 = dmf.ketx0_state()

    st1 = dmf.reduce(np.kron, n_qubits * [st1 @ np.conjugate(st1.T)])

    rho = qt.Qobj(st1, dims=[n_qubits * [2], n_qubits * [2]])

    assert np.array_equal(st1, rho)

    assert dmf.negativity(st1, 4, 4) < 0.1

    graph1 = nx.Graph([(0, 1), (1, 2)])
    rho_all = sconverter.graph_to_density(graph1)

    rho01 = dmf.project_to_z0_and_remove(rho_all, [0, 0, 1])
    assert dmf.negativity(rho01, 2, 2) > 0.1

    rho02 = dmf.project_to_z0_and_remove(rho_all, [0, 1, 0])

    assert dmf.negativity(rho02, 2, 2) < 0.1

    rho12 = dmf.project_to_z0_and_remove(rho_all, [1, 0, 0])
    assert dmf.negativity(rho12, 2, 2) > 0.1


def test_density_to_graph():
    graph1 = nx.Graph([(0, 1), (1, 2)])
    rho_all = sconverter.graph_to_density(graph1)

    rgraph = sconverter.density_to_graph(rho_all)

    assert np.array_equal(nx.to_numpy_matrix(graph1), rgraph)
