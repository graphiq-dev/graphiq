"""
Graph representation visualization
"""
import networkx as nx
import matplotlib.pyplot as plt


def draw_graph(state_graph, show=False):
    """
    It allows one to draw the underlying networkX graph with matplotlib library.
    """
    nx.draw(state_graph.get_graph_id_form(), with_labels=True, font_weight='bold')
    if show:
        plt.show()
