"""
Example of constructing a graph representation and visualizing using matplotlib
"""
from src.backends.graph.state import Graph
from src.visualizers.graph import *

graph1 = nx.Graph([(1, 2), (2, 3), (1, 4), (1, 5), (3, 6), (3, 7)])
graph_rep1 = Graph(data=graph1, root_node_id=2)
graph_rep1.draw()
graph_rep1.add_node(8)
graph_rep1.add_edge(7, 8)
graph_rep1.draw()
