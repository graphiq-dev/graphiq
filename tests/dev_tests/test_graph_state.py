from src.backends.graph.state import Graph
from src.visualizers.graph import *

graph1 = nx.Graph([(1, 2), (2, 3)])
my_graph_rep = Graph(data=graph1, root_node_id=1)
print(my_graph_rep.get_nodes_id_form())
print(my_graph_rep.get_edges_id_form())
# my_graph_rep.add_edge(3, 4)
# print(my_graph_rep.get_edges_id_form())
my_graph_rep.add_node(4)
my_graph_rep.add_edge(3, 4)
print(my_graph_rep.get_edges_id_form())
draw_graph(my_graph_rep)
plt.show()
