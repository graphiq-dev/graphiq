"""
This file contains helper functions for graph representation
"""
import networkx as nx


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
        # graph_data is a single integer, meaning the graph contains only a single node
        # To maintain a consistent ID representation across all nodes, we cast the int to a frozenset
        graph_data = frozenset([graph_data])
        only_node = QuNode(graph_data)
        node_dict[graph_data] = only_node
        graph.add_node(only_node)

    elif isinstance(graph_data, frozenset):
        # graph_data is a single frozenset, meaning the graph contains only a single (redundantly encoded) node
        only_node = QuNode(graph_data)
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
                tmp_node = QuNode(node)
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


class QuNode:
    """
    A class that represents a node of qubit(s). Only simple redundancy encoding is allowed.
    No other QECC is allowed.
    """

    def __init__(self, id_set):
        """
        Creates a node of qubits

        :param id_set: id for the QuNode (if the QuNode has a single qubit/no redundat encoding) OR
                       id for each qubit of the redundantly encoded QuNode
        :type id_set: frozenset OR int
        :raises ValueError: if the wrong datatype is passed in as id_set
        :return: nothing
        :rtype: None
        """
        if isinstance(id_set, frozenset):
            self.id = id_set
        elif isinstance(id_set, int):
            self.id = frozenset([id_set])
        else:
            raise ValueError("QuNode only accepts frozenset and int as id.")

    def count_redundancy(self):
        """
        Return the number of qubits in the redundancy encoding

        :return: the number of qubits in the redundant encoding
        :rtype: int
        """
        return len(self.id)

    def set_id(self, id_set):
        """
        Allow one to update the IDs of all qubits in the node.

        :param id_set: the new set of ids for the qubits in the node
        :type id_set: frozenset OR int
        :raises ValueError: if id_set is not the desired datatype
        :return: function returns nothing
        :rtype: None
        """
        if isinstance(id_set, frozenset):
            self.id = id_set
        elif isinstance(id_set, int):
            self.id = frozenset([id_set])
        else:
            raise ValueError("QuNode only accepts frozenset and int as id.")

    def remove_id(self, photon_id):
        """
        Remove the qubit with the specified id from a redundancy encoding.
        It does nothing if the node is not redundantly encoded.

        :param photon_id: id of the qubit to remove from the redundancy encoding
        :type photon_id: int
        :return: True if the photon of the given ID was removed, False otherwise
        :rtype: bool
        """
        if len(self.id) > 1 and photon_id in self.id:
            tmp_set = set(self.id)
            tmp_set.remove(photon_id)
            self.id = frozenset(tmp_set)
            return True

        return False

    def remove_first_id(self):
        """
        Remove the first qubit from the redundancy encoding.
        It does nothing if the node is not redundantly encoded.

        :return: True if a qubit is removed, False otherwise
        :rtype: bool
        """
        if len(self.id) > 1:
            tmp_set = set(self.id)
            tmp_set.pop()
            self.id = frozenset(tmp_set)
            return True

        return False

    def get_id(self):
        """
        Return the id of the node. This may be either an integer ID
        or a frozenset containing all photon IDs in this node

        :return: the photon(s) id(s)
        :rtype: frozenset
        """
        return self.id
