import numpy as np
import networkx as nx

from functools import reduce
from src.backends.density_matrix.state import DensityMatrix
from src.backends.graph.state import Graph
import src.backends.density_matrix.functions as dmf
import src.backends.graph.functions as gf
import src.backends.stabilizer.functions as sf

import src.backends.state_representation_conversion as sconverter


def test_negativity():
    n_qubits = 4

    st1 = dmf.ketx0_state()

    st1 = dmf.reduce(np.kron, n_qubits * [st1 @ np.conjugate(st1.T)])

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

    assert np.allclose(nx.to_numpy_matrix(graph1), rgraph)

def test_density_to_graph2():
    graph1 = nx.Graph([(0, 1), (1, 2), (1, 3)])
    rho_all = sconverter.graph_to_density(graph1)

    rgraph = sconverter.density_to_graph(rho_all)

    assert np.allclose(nx.to_numpy_matrix(graph1), rgraph)


def test_stabilizer_to_graph():
    x_matrix = np.eye(4)
    graph1 = nx.Graph([(1, 2), (2, 3), (3, 4)])
    z_matrix = nx.to_numpy_matrix(graph1)
    generator_list = sf.symplectic_to_string(x_matrix, z_matrix)
    graph2 = sconverter.stabilizer_to_graph(generator_list)
    assert nx.is_isomorphic(graph1, graph2)


def test_stabilizer_and_density_conversion():
    x_matrix = np.eye(4)
    graph1 = nx.Graph([(1, 2), (2, 3), (3, 4)])
    z_matrix = nx.to_numpy_matrix(graph1)
    generator_list = sf.symplectic_to_string(x_matrix, z_matrix)
    rho1 = sconverter.stabilizer_to_density(generator_list)
    rho2 = sconverter.graph_to_density(graph1)

    assert np.allclose(rho1, rho2)

    stabilizer1 = sconverter.density_to_stabilizer(rho2)
    stabilizer2 = sconverter.graph_to_stabilizer(graph1)

    assert reduce(lambda x, y: x and y,
                  map(lambda a, b: a == b, stabilizer1,  stabilizer2), True)


def test_stabilizer_and_density_conversion2():
    x_matrix = np.eye(5)
    graph1 = nx.Graph([(1, 2), (2, 3), (3, 4), (3, 5)])
    z_matrix = nx.to_numpy_matrix(graph1)
    generator_list = sf.symplectic_to_string(x_matrix, z_matrix)
    rho1 = sconverter.stabilizer_to_density(generator_list)
    rho2 = sconverter.graph_to_density(graph1)

    assert np.allclose(rho1, rho2)

    stabilizer1 = sconverter.density_to_stabilizer(rho2)
    stabilizer2 = sconverter.graph_to_stabilizer(graph1)

    assert reduce(lambda x, y: x and y,
                  map(lambda a, b: a == b, stabilizer1,  stabilizer2), True)

