"""
This file contains helper functions for graph representation
"""
import networkx as nx

def convert_data_to_graph(graph_data,root_id):
    """
    A helper function for graph constructor.
    This function accepts multiple input formats and constructs a networkX graph where each node is a QuNode object.
    :params root_id: index for the root qubit (qudit) without prefix
    :params graph_data: data for the graph
    """
    # convert a list of edges to graph using input structure like networkx
    graph = nx.Graph()
    node_dict = dict()
    if isinstance(graph_data,int):
        # graph_data is a single integer, meaning the graph contains only a single node
        tmp_node = QuNode(graph_data)
        node_dict[graph_data]= tmp_node
        graph.add_node(tmp_node)
        root_node = tmp_node
    elif isinstance(graph_data,frozenset):
        # graph_data is a single frozenset, meaning the graph contains only a single (redundantly encoded) node

        tmp_node = QuNode(graph_data)
        node_dict[graph_data]= tmp_node
        graph.add_node(tmp_node)
        root_node = tmp_node
    elif isinstance(graph_data,nx.Graph):
        # graph_data is itself a networkX Graph object.
        for node in graph_data.nodes():
            if isinstance(node, QuNode):
                id = node.get_id()
                node_dict[id] = node
                node.set_id(id)
                graph.add_node(node)
            elif isinstance(node,frozenset) or isinstance(node,int):

                tmp_node = QuNode(node)
                node_dict[node] = tmp_node
                graph.add_node(tmp_node)
            else:
                raise ValueError('Data type in the graph is not supported.')

        for data_pair in graph_data.edges():
            if isinstance(data_pair[0],QuNode):
                graph.add_edge(node_dict[data_pair[0].get_id()],node_dict[data_pair[1].get_id()])
            elif (isinstance(data_pair[0],int) or isinstance(data_pair[0],frozenset)) and  (isinstance(data_pair[1],int) or isinstance(data_pair[1],frozenset)):
                graph.add_edge(node_dict[data_pair[0]],node_dict[data_pair[1]])
            else:
                raise ValueError("Edges contain invalid data type.")
        root_node = node_dict[root_id]
    else:
        if len(graph_data) == 0:
            root_node = None
        else:
            for data_pair in graph_data:
                # data_pair is a pair of vertices in a tuple
                # first add vertices if not existed
                if data_pair[0] not in node_dict.keys():

                    tmp_node = QuNode(data_pair[0])
                    node_dict[data_pair[0]] = tmp_node
                    graph.add_node(tmp_node)
                if data_pair[1] not in node_dict.keys():
                    tmp_node = QuNode(data_pair[1])
                    node_dict[data_pair[1]] = tmp_node
                    graph.add_node(tmp_node)
                # add the edge
                graph.add_edge(node_dict[data_pair[0]],node_dict[data_pair[1]])
            root_node = node_dict[root_id]

    return root_node, node_dict, graph


""""
A class that represents a node of qubit(s). Only simple redundancy encoding is allowed.
No other QECC is allowed.
"""
class QuNode():

    def __init__(self, id_set):
        if isinstance(id_set,frozenset) or isinstance(id_set, int):
            self.id = id_set
        else:
            raise ValueError('QuNode only accepts frozenset and int as id.')

    def count_redundancy(self):
        """
        Return the number of qubits in a redundancy encoding
        """
        if isinstance(self.id,frozenset):
            return len(self.id)
        else:
            return 1

    def set_id(self,id_set):
        """
        Allow one to update the IDs of all qubits in the node.
        """
        if isinstance(id_set,frozenset) or isinstance(id_set, int):
            self.id = id_set
        else:
            raise ValueError('QuNode only accepts frozenset and int as id.')


    def remove_id(self,id):
        """
        Remove the qubit with the specified id from a redudancy encoding.
        It does nothing if the node is not redundantly encoded.
        """
        if isinstance(id, frozenset):
            if id in self.id:
                tmp_set = set(self.id)
                tmp_set.remove(id)
                self.id = frozenset(tmp_set)

    def remove_first_id(self):
        """
        Remove the first qubit from the redundancy encoding.
        It does nothing if the node is not redundantly encoded.
        """
        if isinstance(self.id, frozenset):
            tmp_set = set(self.id)
            tmp_set.pop()
            self.id = frozenset(tmp_set)

    def get_id(self):
        """
        Return the id of the node
        """
        return self.id
