"""

"""
import networkx as nx
import warnings

from src.backends.state_base import StateRepresentationBase
import src.backends.graph.functions as gf
from src.backends.graph.functions import QuNode


class Graph(StateRepresentationBase):
    """
    Graph representation of a graph state.
    As the intermediate states of the process may not be graph states (but assuming still stabilizer states),
    we may need to keep track of local Cliffords that convert the state to the graph state represented by the graph
    """

    def __init__(self, data, root_node_id, *args, **kwargs):
        """
        :params state_data: data used to construct the representation
        :params state_id: a unique identifier for the state
        :params root_node_id: a node id for the root node
        """

        super().__init__(data, *args, **kwargs)
        self.root_node, self.node_dict, self.data = gf.convert_data_to_graph(data, root_node_id)
        self.local_cliffords = None  # set this later

    def add_node(self, node_to_add):
        """
        Add a node to the graph.
        It allows node_to_add to be one of following types:
            QuNode
            int
            frozenset
        """
        if isinstance(node_to_add, QuNode):
            node_id = node_to_add.get_id()
            if node_id not in self.node_dict.keys():
                self.node_dict[node_id] = node_to_add
                self.data.add_node(node_to_add)
            else:
                warnings.warn('Node already in the graph. Check node identifier.')
        elif isinstance(node_to_add, int) or isinstance(node_to_add, frozenset):
            # node_to_add is just a node id; create the node first if it does not exist
            if node_to_add not in self.node_dict.keys():
                tmp_node = QuNode(node_to_add)
                self.node_dict[node_to_add] = tmp_node
                self.data.add_node(tmp_node)
            else:
                warnings.warn('Node already in the graph. Check node identifier.')
        else:
            # invalid data Type
            raise ValueError('Invalid data for the node to be added.')

    def add_edge(self, first_node, second_node):
        """
        Add an edge between two nodes. If any of these two nodes does not exist, no edge is added.
        """
        if isinstance(first_node, QuNode):
            node_id1 = first_node.get_id()
        elif isinstance(first_node, int) or isinstance(first_node, frozenset):
            node_id1 = first_node
        else:
            raise ValueError('Not supporting input data type')
        if isinstance(second_node, QuNode):
            node_id2 = second_node.get_id()
        elif isinstance(second_node, int) or isinstance(second_node, frozenset):
            node_id2 = second_node
        else:
            raise ValueError('Not supporting input data type')

        if node_id1 in self.node_dict.keys() and node_id2 in self.node_dict.keys():
            self.data.add_edge(self.node_dict[node_id1], self.node_dict[node_id2])
        else:
            warnings.warn('At least one of nodes do not exist. Not adding an edge.')

    def get_edges(self):
        return [e for e in self.data.edges]

    def get_nodes(self):
        return list(self.node_dict.values())

    def get_node_by_id(self,node_id):
        if node_id in self.node_dict:
            return self.node_dict[node_id]
        else:
            return None

    def get_edges_id_form(self):
        return [(e[0].get_id(),e[1].get_id()) for e in self.data.edges]

    def get_nodes_id_form(self):
        return [node.get_id() for node in self.data.nodes]

    def get_root_node(self):
        return self.root_node

    def get_node_dict(self):
        return self.node_dict

    def n_node(self):
        return len(self.node_dict.keys())

    def n_qubit(self):
        number_qubit = 0
        for node_id in self.node_dict.keys():
            if isinstance(node_id, frozenset):
                number_qubit += len(frozenset)
            else:
                number_qubit += 1
        return number_qubit

    def n_redundant_encoding_node(self):
        number_redundant_node = 0
        for node_id in self.node_dict.keys():
            if isinstance(node_id, frozenset):
                number_redundant_node += 1
        return number_redundant_node

    def get_graph_id_form(self):
        tmp_graph = nx.Graph(self.get_edges_id_form())
        nodelist = self.get_nodes_id_form()
        if set(nodelist) != set(tmp_graph.nodes()):
            for node in nodelist:
                tmp_graph.add_node(node)
        return tmp_graph

    def get_neighbors(self, node_id):
        """
        Return the list of all neighbors of the node with node_id
        """
        neighbor_list = list()
        if node_id in self.node_dict.keys():
            cnode = self.node_dict[node_id]
            all_nodes = self.node_dict.values()

            for node in all_nodes:
                if (node,cnode) in self.data.edges() or (cnode, node) in self.data.edges():
                    neighbor_list.append(node)
        return neighbor_list

    def remove_node(self, node_id):
        """
        Remove a node from the graph and remove all edges of the node
        """
        if node_id in self.node_dict.keys():
            self.data.remove_node(self.node_dict[node_id])
        else:
            warnings.warn('No node is removed since node id does not exist.')

    def remove_id_from_redundancy(self, node_id, removal_id=None):
        """
        Remove a photon from the redunantly encoded node.
        If no photon id is specified, then the first photon in the node is removed.
        """
        if node_id in self.node_dict.keys():
            if isinstance(node_id, frozenset):
                node = self.get_node_by_id(node_id)
                if removal_id is None:
                    node.remove_first_id()
                else:
                    node.remove_id(removal_id)
                new_node_id = node.get_id()
                if node.count_redundancy() == 0:
                    self.data.remove_node(self.node_dict[node_id])
                    self.node_dict.pop(node_id)
                else:
                    self.node_dict[new_node_id] = self.node_dict.pop(node_id)

            else:
                warnings.warn('No redundancy is removed since this node is not redundantly encoded.')
        else:
            warnings.warn('No node is removed since node id does not exist.')

    def measureX(self, node_id):
        """
        Measure a given node in the X basis.
        If the node contains a single photon, this measurement removes the node and disconnects all edges of this node.
        If the node is redunantly encoded, this measurement removes one photon from the node.
        """
        if node_id in self.node_dict.keys():
            cnode = self.node_dict[node_id]
            if cnode.count_redundancy() == 1:
                self.data.remove_node(cnode)
            else:
                new_node_id = set(node_id)
                cnode.remove_id(new_node_id.pop())
                self.node_dict[frozenset(new_node_id)] = self.node_dict.pop(node_id)
        else:
            warnings.warn('No action is applied since node id does not exist.')

    def measureY(self, node_id):
        # TODO
        raise NotImplementedError('To do')

    def measureZ(self, node_id):
        # TODO
        raise NotImplementedError('To do')

    def draw(self):
        """
        It allows one to draw the underlying networkX graph with matplotlib library.
        """
        nx.draw(self.get_graph_id_form(), with_labels=True, font_weight='bold')


