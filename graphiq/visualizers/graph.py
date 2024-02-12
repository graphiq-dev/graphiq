"""
Graph representation visualization helper
"""
import matplotlib.pyplot as plt
import networkx as nx


def draw_graph(state_graph, show=False, ax=None, with_labels=True):
    """
    Allows one to draw the underlying networkX graph with matplotlib library.

    :param state_graph: Graph state representation which we want to plot
    :type state_graph: Graph
    :param show: If True, draw the graph and show it. Otherwise, draw but do not show
    :type show: bool
    :param ax: axis on which to draw the graph
    :type ax: matplotlib.Axis
    :param with_labels: True if drawing the labels; False otherwise
    :type with_labels: bool
    :return: fig, axes on which the state is drawn
    :rtype: matplotlib.Figure, matplotlib.Axes
    """

    nx.draw(
        state_graph.data,
        with_labels=with_labels,
        font_weight="bold",
        ax=ax,
    )
    if show:
        plt.show()
    fig = plt.gcf()
    ax = plt.gca()
    return fig, ax
