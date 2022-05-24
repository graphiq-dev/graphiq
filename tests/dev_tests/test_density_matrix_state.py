import src.backends.density_matrix.functions
from src.backends.density_matrix.state import DensityMatrix
import numpy as np
import networkx as nx

# test creating a density matrix with both an np array and a nx graph
state_rep1 = DensityMatrix(np.eye(2))
graph1 = nx.Graph([(1, 2), (1, 3)])
state_rep2 = DensityMatrix.from_graph(graph1)

print(state_rep1.data)
print(state_rep2.data)
print(src.backends.density_matrix.functions.is_psd(state_rep2.data))
print(src.backends.density_matrix.functions.get_single_qubit_gate(3, 0, src.backends.density_matrix.functions.sigmaz()))
print(src.backends.density_matrix.functions.get_controlled_gate(3, 0, 1, src.backends.density_matrix.functions.sigmax()))
