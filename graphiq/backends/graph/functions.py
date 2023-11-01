"""
This file contains helper functions for graph representation
"""
import networkx as nx

from graphiq.backends.graph.node import QuNode


def convert_data_to_graph(graph_data):
    """
    A helper function for graph constructor.
    This function accepts multiple input formats and constructs a networkX graph where each node is a QuNode object.

    :param graph_data: data for the graph
    :type graph_data: frozenset OR int OR networkx.Graph OR iterable of data pairs
    :raises ValueError: if graph_data is not a supported data type
    :return: node dictionary (keys = node ids, value = QuNode objects), graph (graph constructed from data)
    :rtype: dict, networkx.Graph
    """
    # convert a list of edges to graph using input structure like networkx
    graph = nx.Graph()
    node_dict = dict()

    if isinstance(graph_data, int):
        # graph_data is a single integer, which specifies the number of nodes in the graph
        # To maintain a consistent ID representation across all nodes, we cast the int to a frozenset
        for i in range(graph_data):
            graph_data = frozenset([i])
            only_node = QuNode(graph_data)
            node_dict[graph_data] = only_node
            graph.add_node(only_node)

    elif isinstance(graph_data, frozenset):
        # graph_data is a single frozenset, meaning the graph contains only a single (redundantly encoded) node
        if len(graph_data) > 1:
            redundancy = True
        else:
            redundancy = False
        only_node = QuNode(graph_data, redundancy=redundancy)
        node_dict[graph_data] = only_node
        graph.add_node(only_node)

    elif isinstance(graph_data, nx.Graph):
        # graph_data is itself a networkX Graph object.
        for node in graph_data.nodes():
            if isinstance(node, QuNode):
                node_id = node.get_id()
                node_dict[node_id] = node
                node.set_id(node_id)
                graph.add_node(node)
            elif isinstance(node, frozenset) or isinstance(node, int):
                if isinstance(node, int):
                    node = frozenset([node])  # cast int to frozenset
                if len(node) > 1:
                    redundancy = True
                else:
                    redundancy = False
                tmp_node = QuNode(node, redundancy=redundancy)
                node_dict[node] = tmp_node
                graph.add_node(tmp_node)
            else:
                raise ValueError("Data type in the graph is not supported.")

        for data_pair in graph_data.edges():
            if isinstance(data_pair[0], QuNode):
                graph.add_edge(
                    node_dict[data_pair[0].get_id()], node_dict[data_pair[1].get_id()]
                )
            elif (
                    isinstance(data_pair[0], int) or isinstance(data_pair[0], frozenset)
            ) and (
                    isinstance(data_pair[1], int) or isinstance(data_pair[1], frozenset)
            ):
                # Cast ints to frozensets if necessary
                new_data_pair = [data_pair[0], data_pair[1]]
                if isinstance(data_pair[0], int):
                    new_data_pair[0] = frozenset([data_pair[0]])
                if isinstance(data_pair[1], int):
                    new_data_pair[1] = frozenset([data_pair[1]])
                graph.add_edge(node_dict[new_data_pair[0]], node_dict[new_data_pair[1]])
            else:
                raise ValueError("Edges contain invalid data type.")

    else:

        for data_pair in graph_data:
            # data_pair is a pair of vertices in a tuple

            # First, cast any ints to frozensets
            new_data_pair = [data_pair[0], data_pair[1]]
            if isinstance(data_pair[0], int):
                new_data_pair[0] = frozenset([data_pair[0]])
            if isinstance(data_pair[1], int):
                new_data_pair[1] = frozenset([data_pair[1]])

            # then add vertices if not existed
            if new_data_pair[0] not in node_dict.keys():
                tmp_node = QuNode(new_data_pair[0])
                node_dict[new_data_pair[0]] = tmp_node
                graph.add_node(tmp_node)
            if new_data_pair[1] not in node_dict.keys():
                tmp_node = QuNode(new_data_pair[1])
                node_dict[new_data_pair[1]] = tmp_node
                graph.add_node(tmp_node)
            # add the edge
            graph.add_edge(node_dict[new_data_pair[0]], node_dict[new_data_pair[1]])

    return node_dict, graph
