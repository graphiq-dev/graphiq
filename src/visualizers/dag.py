"""
Functions for visualizing the circuit DAGs
"""

import networkx as nx


def dag_topology_pos(dag, method="topology"):
    """
    Returns the node positions for plotting a DAG structure

    The methods are:
        topology: uses the requirement of being a DAG to order nodes from left to right in the given topological order
        spring: uses the default networkx spring layout option

    :param dag:
    :param method:
    :return:
    """
    if method == "topology":
        """Display in topological order, with simple offsetting for legibility"""
        pos_dict = {}
        for i, node_list in enumerate(nx.topological_generations(dag)):
            y_offset = len(node_list) / 2
            x_offset = 0.1
            for j, name in enumerate(node_list):
                pos_dict[name] = (i - j * x_offset, j - y_offset)

    elif method == "spring":
        pos_dict = nx.spring_layout(dag, seed=0)  # Seed layout for reproducibility

    else:
        pos_dict = nx.spring_layout(dag, seed=0)  # Seed layout for reproducibility

    return pos_dict
