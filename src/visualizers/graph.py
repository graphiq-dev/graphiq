"""
Graph representation visualization
"""
import networkx as nx
import matplotlib.pyplot as plt

# TODO: change all plotting functions to have the same API (you can pass in axes, will return ax/fig otherwise)


def draw_graph(state_graph, show=False, ax=None):
    """
    It allows one to draw the underlying networkX graph with matplotlib library.
    """
    nx.draw(state_graph.get_graph_id_form(), with_labels=True, font_weight='bold', ax=ax)
    if show:
        plt.show()
