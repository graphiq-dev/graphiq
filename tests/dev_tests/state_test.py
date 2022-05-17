from src.backends.state_representations import *
import numpy as np
import networkx as nx


state_rep1 = DensityMatrix(np.eye(2))

graph1 = nx.Graph([(1,2),(1,3)])

state_rep2 = DensityMatrix(graph1)


print(state_rep1.get_rep())
print(state_rep2.get_rep())
print(is_psd(state_rep2.get_rep()))
print(get_single_qubit_gate(3,0,sigmaz()))
print(get_controlled_gate(3,0,1,sigmax()))
