import src.backends.state_rep_conversion as rc
from src.backends.density_matrix.state import DensityMatrix
from src.backends.density_matrix.compiler import DensityMatrixCompiler
import src.backends.density_matrix.functions as dmf
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


def test_y_measurement():
    graph1 = nx.Graph([(1, 2), (2, 3), (3, 4), (1, 3)])
    rho = DensityMatrix(rc.graph_to_density(graph1))
    kety0 = dmf.state_kety0()
    kety1 = dmf.state_kety1()
    projector0 = dmf.get_one_qubit_gate(4, 2, kety0 @ np.conjugate(kety0.T))
    projector1 = dmf.get_one_qubit_gate(4, 2, kety1 @ np.conjugate(kety1.T))
    projectors = [projector0, projector1]
    rho.apply_measurement(projectors, measurement_determinism=0)
    graph_adj = rc.density_to_graph(rho.data)
    nx.draw(nx.from_numpy_array(graph_adj), with_labels=True)
    plt.show()


def test_x_measurement():
    graph1 = nx.Graph([(1, 2), (2, 3), (3, 4), (1, 3), (1, 4), (2, 4), (1, 5), (2, 5)])
    # nx.draw(graph1, with_labels=True)
    rho = DensityMatrix(rc.graph_to_density(graph1))
    ketx0 = dmf.state_ketx0()
    ketx1 = dmf.state_ketx1()
    graph2 = nx.Graph([(1, 2), (1, 4), (2, 4), (1, 5), (2, 5)])
    graph2.add_node(3)
    nx.draw(graph2, with_labels=True)
    rho2 = DensityMatrix(rc.graph_to_density(graph2))
    projector0 = dmf.get_one_qubit_gate(5, 2, ketx0 @ np.conjugate(ketx0.T))
    projector1 = dmf.get_one_qubit_gate(5, 2, ketx1 @ np.conjugate(ketx1.T))
    projectors = [projector0, projector1]
    rho.apply_measurement(projectors, measurement_determinism=1)
    print(f"rho is {rho.data}")
    print(f"rho2 is {rho2.data}")

    graph_adj = rc.density_to_graph(rho.data)
    # nx.draw(nx.from_numpy_array(graph_adj),with_labels=True)
    plt.show()
