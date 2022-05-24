from src.states import *
import networkx as nx

graph1 = nx.Graph([(1,2),(2,3)])
my_graph_rep = GraphState(data=graph1, root_node_id=1)
print(my_graph_rep.get_nodes_id_form())
print(my_graph_rep.get_edges_id_form())
my_graph_rep.add_edge(3,4)
print(my_graph_rep.get_edges_id_form())
my_graph_rep.add_node(4)
my_graph_rep.add_edge(3,4)
print(my_graph_rep.get_edges_id_form())
