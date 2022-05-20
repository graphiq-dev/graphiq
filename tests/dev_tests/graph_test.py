from src.backends.state_representations import *
import src.backends.density_matrix.functions as dmf
import src.backends.graph.functions as gf
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt



graph1 = nx.Graph([(1,2),(2,3)])
my_graph_rep = GraphRep(state_data=graph1, root_node_id=1)
print(my_graph_rep.get_nodes_id_form())
print(my_graph_rep.get_edges_id_form())
my_graph_rep.add_edge(3,4)
print(my_graph_rep.get_edges_id_form())
my_graph_rep.add_node(4)
my_graph_rep.add_edge(3,4)
print(my_graph_rep.get_edges_id_form())
