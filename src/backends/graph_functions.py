"""
This file contains helper functions for graph representation
"""
import networkx as nx

def convert_data_to_graph(root_id, initial_counter, graph_data):
    """
    A helper function for graph constructor.
    This function accepts multiple input formats and constructs a networkX graph where each node is a QuNode object.
    :params root_id: index for the root qubit (qudit) without prefix
    :params initial_counter: a prefix for the id
    :params graph_data: data for the graph
    """
    # convert a list of edges to graph using input structure like networkx
    graph = nx.Graph()
    tmp_dict = dict()
    if isinstance(graph_data,int):
        # graph_data is a single integer, meaning the graph contains only a single node
        tmp_node = QuNode(initial_counter + graph_data)
        tmp_dict[graph_data]= tmp_node
        graph.add_node(tmp_node)
        root_node = tmp_node
    elif isinstance(graph_data,frozenset):
        # graph_data is a single frozenset, meaning the graph contains only a single (redundantly encoded) node
        new_id = [id + initial_counter for id in graph_data]
        new_id = frozenset(new_id)
        tmp_node = QuNode(new_id)
        tmp_dict[new_id]= tmp_node
        graph.add_node(tmp_node)
        root_node = tmp_node
    elif isinstance(graph_data,nx.Graph):
        # graph_data is itself a networkX Graph object.
        for node in graph_data.nodes():
            if isinstance(node, QuNode):
                id = node.get_id()
                tmp_dict[id] = node
                node.set_id_with_prefix(initial_counter,id)
                graph.add_node(node)
            else:
                if isinstance(node,frozenset):
                    new_id = [id+initial_counter for id in node]
                    new_id = frozenset(new_id)
                    tmp_node = QuNode(new_id)
                else:
                    tmp_node = QuNode(initial_counter+node)
                tmp_dict[node] = tmp_node
                graph.add_node(tmp_node)
        for data_pair in graph_data.edges():
            if isinstance(data_pair[0],QuNode):
                graph.add_edge(tmp_dict[data_pair[0].get_id()],tmp_dict[data_pair[1].get_id()])
            elif (isinstance(data_pair[0],int) or isinstance(data_pair[0],frozenset)) and  (isinstance(data_pair[1],int) or isinstance(data_pair[1],frozenset)):
                graph.add_edge(tmp_dict[data_pair[0]],tmp_dict[data_pair[1]])
            else:
                raise ValueError("Edges contain invalid data type.")
        root_node = tmp_dict[root_id]
    else:
        if len(graph_data) == 0:
            root_node = None
        else:
            for data_pair in graph_data:
                # data_pair is a pair of vertices in a tuple
                # first add vertices if not existed
                if data_pair[0] not in tmp_dict.keys():
                    if isinstance(data_pair[0],frozenset):
                        new_id = [id+initial_counter for id in data_pair[0]]
                        new_id = frozenset(new_id)
                        tmp_node = QuNode(new_id)
                    elif isinstance(data_pair[0],int):
                        tmp_node = QuNode(data_pair[0]+initial_counter)
                    tmp_dict[data_pair[0]] = tmp_node
                    graph.add_node(tmp_node)
                if data_pair[1] not in tmp_dict.keys():
                    if isinstance(data_pair[1],frozenset):
                        new_id = [id+initial_counter for id in data_pair[1]]
                        new_id = frozenset(new_id)
                        tmp_node = QuNode(new_id)
                    elif isinstance(data_pair[1],int):
                        tmp_node = QuNode(data_pair[1]+initial_counter)
                    tmp_dict[data_pair[1]] = tmp_node
                    graph.add_node(tmp_node)
                # add the edge
                graph.add_edge(tmp_dict[data_pair[0]],tmp_dict[data_pair[1]])
            root_node = tmp_dict[root_id]

    return root_node, tmp_dict, graph


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

    def set_id_with_prefix(self, initial_counter, id_set):
        """
        Allow one to update the IDs of all qubits in the node.
        The id format is state id + qubit id.
        The motivation is to have a unique id for each qubit in the process.
        Initial_counter is supposed to be the state id.
        """
        if isinstance(id_set,frozenset):
            tmp_id_list = list(id_set)
            tmp_id_list2 = [initial_counter + id for id in tmp_id_list]
            self.id = frozenset(tmp_id_list2)
        elif isinstance(id_set, int):
            self.id = initial_counter + id_set
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

    def get_id_wo_prefix(self,initial_counter):
        """
        Return the id of the node by substracting initial_counter from the id
        """
        if isinstance(self.id,frozenset):
            tmp_id_list = list(self.id)
            tmp_id_list2 = [id - initial_counter for id in tmp_id_list]
            return frozenset(tmp_id_list2)
        else:
            return self.id - initial_counter
