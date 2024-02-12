"""
Graph representation of quantum state

"""

import itertools
import warnings
from collections.abc import Iterable

# TODO: Refactor and revise when building the compiler for graph representation
import networkx as nx
import numpy as np

import graphiq.backends.graph.functions as gf
from graphiq.backends.graph.node import QuNode
from graphiq.backends.lc_equivalence_check import is_lc_equivalent
from graphiq.backends.state_base import StateRepresentationBase
from graphiq.visualizers.graph import draw_graph


class Graph(StateRepresentationBase):
    """
    Graph representation of a graph state.
    As the intermediate states of the process may not be graph states (but assuming still stabilizer states),
    we may need to keep track of local Clifford gates that convert the state to the graph state represented by the graph

    """

    def __init__(self, data, *args, **kwargs):
        """
        Create a Graph representation object

        :param data: data used to construct the representation
        :type data: frozenset OR int OR networkx.Graph OR iterable of data pairs
        :return: function returns nothing
        :rtype: None
        """

        super().__init__(data, *args, **kwargs)
        self.node_dict, self.data = gf.convert_data_to_graph(data)

    def add_node(self, node_to_add):
        """
        Add a node to the graph.
        It allows node_to_add to be one of following types:

            * QuNode
            * int
            * frozenset

        :param node_to_add: node id to add to the Graph representation
        :type node_to_add: QuNode OR int OR frozenset
        :raises ValueError: if node_to_add is of an invalid datatype
        :return: function returns nothing
        :rtype: None
        """
        if isinstance(node_to_add, QuNode):
            node_id = node_to_add.get_id()
            if node_id not in self.node_dict.keys():
                self.node_dict[node_id] = node_to_add
                self.data.add_node(node_to_add)
            else:
                warnings.warn("Node already in the graph. Check node identifier.")
        elif isinstance(node_to_add, int) or isinstance(node_to_add, frozenset):
            if isinstance(node_to_add, int):
                node_to_add = frozenset([node_to_add])
            # node_to_add is just a node id; create the node first if it does not exist
            if node_to_add not in self.node_dict.keys():
                tmp_node = QuNode(node_to_add)
                self.node_dict[node_to_add] = tmp_node
                self.data.add_node(tmp_node)
            else:
                warnings.warn("Node already in the graph. Check node identifier.")
        else:
            raise ValueError("Invalid data for the node to be added.")

    def add_edge(self, first_node, second_node):
        """
        Add an edge between two nodes. If any of these two nodes does not exist, no edge is added.

        :param first_node: the first node on which to add an edge
        :type first_node: QuNode OR int OR frozenset
        :param second_node: the second node on which to add an edge
        :type second_node: QuNode OR int OR frozenset
        :return: nothing
        :rtype: None
        """
        if isinstance(first_node, QuNode):
            node_id1 = first_node.get_id()
        elif isinstance(first_node, int) or isinstance(first_node, frozenset):
            if isinstance(first_node, int):
                first_node = frozenset([first_node])
            node_id1 = first_node
        else:
            raise ValueError("Not supporting input data type")

        if isinstance(second_node, QuNode):
            node_id2 = second_node.get_id()
        elif isinstance(second_node, int) or isinstance(second_node, frozenset):
            if isinstance(second_node, int):
                second_node = frozenset([second_node])
            node_id2 = second_node
        else:
            raise ValueError("Not supporting input data type")

        if node_id1 in self.node_dict.keys() and node_id2 in self.node_dict.keys():
            self.data.add_edge(self.node_dict[node_id1], self.node_dict[node_id2])
        else:
            warnings.warn("At least one of nodes do not exist. Not adding an edge.")

    def get_edges(self):
        """
        Get all graph edges (entangled pairs) in the Graph state representation

        :return: graph edges
        :rtype: list
        """
        return [e for e in self.data.edges]

    def get_nodes(self):
        """
        Get all graph nodes (qubits) in the Graph state representation

        :return: all QuNodes in the Graph
        :rtype: list
        """
        return list(self.node_dict.values())

    def get_node_by_id(self, node_id):
        """
        Retrieve a QuNode object by ID

        :param node_id: the ID of the node to retrieve
        :type node_id: frozenset OR int
        :return: the Graph node OR None (if no such ID exists)
        :rtype: QuNode OR None
        """
        if isinstance(node_id, int):
            node_id = frozenset([node_id])

        if node_id in self.node_dict:
            return self.node_dict[node_id]
        else:
            # TODO: determine whether this should raise an error instead
            warnings.warn(f"ID {node_id} does not exist in this Graph representation")
            return None

    def get_edges_id_form(self):
        """
        Returns the list of edges in the Graph as described by tuples of node IDs

        :return: list of tuples of node IDs, corresponding to the graph edges
        :rtype: list[tuples]
        """
        return [(e[0].get_id(), e[1].get_id()) for e in self.data.edges]

    def get_nodes_id_form(self):
        """
        Retrieve a list of all node IDs

        :return: node IDs in the graph
        :rtype: list
        """
        return [node.get_id() for node in self.data.nodes]

    def edge_exists(self, node1, node2):
        """
        Checks whether an edge (parameterized by 2 nodes) exists

        :param node1: First node of potential edge
        :type node1: int, frozenset, or QuNode
        :param node2: Second node of potential edge
        :type node2: int, frozenset, or QuNode
        :return: True if the edge exists, False otherwise
        :rtype: bool
        """
        node1, node2 = self._node_as_qunode(node1), self._node_as_qunode(node2)
        return self.data.has_edge(node1, node2)

    def lc_equivalent(self, other_graph, mode="deterministic"):
        """
        Determines whether two graph states are local-Clifford equivalent or not, given the adjacency matrices of the two.
        It takes two adjacency matrices as input and returns a numpy.array containing :math:`n` (:math:`2 \\times 2` array)s
        = clifford operations on each qubit.

        :param other_graph: the other graph against which to check LC equivalence
        :type other_graph: Graph
        :param mode: the chosen mode for finding solutions. It can be either 'deterministic' (default) or 'random'.
        :type mode: str
        :raises AssertionError: if the number of rows in the row reduced matrix is less than the rank of coefficient
            matrix or if the number of linearly dependent columns is not equal to :math:`4n - rank`
            (for :math:`n` being the number of nodes in the graph)
        :return: If a solution is found, returns True and an array of single-qubit Clifford :math:`2 \\times 2` matrices
            in the symplectic formalism. If not, graphs are not LC equivalent and returns False, None.
        :rtype: bool, numpy.ndarray or None
        """
        g1 = nx.to_numpy_array(self.data).astype(int)
        g2 = nx.to_numpy_array(other_graph.data).astype(int)
        return is_lc_equivalent(g1, g2, mode=mode)

    @property
    def n_node(self):
        """
        Returns the number of nodes in the Graph

        :return: the number of nodes in the Graph
        :rtype: int
        """
        return len(self.node_dict.keys())

    @property
    def n_qubits(self):
        """
        Returns the number of qubits in the graph (counting the redundant encoding as separate qubits)

        :return: the number of qubits
        :rtype: int
        """
        n_qubits = 0
        for node_id in self.node_dict.keys():
            n_qubits += len(node_id)

        return n_qubits

    @property
    def n_redundant_encoding_node(self):
        """
        Number of nodes in the graph which have redundant encoding

        :return: the number of nodes with redundant encoding
        :rtype: int
        """
        number_redundant_node = 0
        for photon_id in self.node_dict.keys():
            if len(photon_id) > 1:
                number_redundant_node += 1
        return number_redundant_node

    def get_graph_id_form(self):
        """
        Get the state Graph where, instead of each node being a QuNode, each node
        is its own QuNode ID

        :return: the state graph, which each QuNode replaced by its ID
        :rtype: nx.Graph
        """
        tmp_graph = nx.Graph(self.get_edges_id_form())
        nodelist = self.get_nodes_id_form()
        if set(nodelist) != set(tmp_graph.nodes()):
            for node in nodelist:
                tmp_graph.add_node(node)

        return tmp_graph

    def get_neighbors(self, node_id):
        """
        Return the list of all neighbors (i.e. nodes connected by an edge) of the node with node_id

        :param node_id: the ID of the node which we want to find the neighbours of
        :type node_id: frozenset OR int
        :return: a list of neighbours for the node with node_id
        :rtype: list
        """
        if isinstance(node_id, int):
            node_id = frozenset([node_id])
        return list(self.data.neighbors(self.node_dict[node_id]))

    def remove_node(self, node_id):
        """
        Remove a node from the Graph representation and remove all edges of the node
        Also update node_dict accordingly

        :param node_id: the ID of the node to remove
        :type node_id: int OR frozenset
        :return: True if the node is successfully removed, False otherwise
        :rtype: bool
        """
        if isinstance(node_id, int):
            node_id = frozenset([node_id])
        if node_id in self.node_dict.keys():
            self.data.remove_node(self.node_dict[node_id])
            self.node_dict.pop(node_id, None)
            return True
        else:
            warnings.warn("No node is removed since node id does not exist.")
            return False

    def remove_edge(self, id1, id2):
        """
        Removes an edge from the graph if it exists

        :param id1: ID of the first node of the edge
        :type id1:  int or frozenset
        :param id2: ID of the second node of the edge
        :type id2: int or frozenset
        :return: True if edge successfully removed, False otherwise
        """
        node1, node2 = self._node_as_qunode(id1), self._node_as_qunode(id2)
        try:
            self.data.remove_edge(node1, node2)
            return True
        except nx.NetworkxError:
            warnings.warn("No edge is removed since edge does not exist.")
            return False

    def node_is_redundant(self, node_id):
        """
        Checks whether or not a given node is redundant (i.e. whether it contains more than 1 photon)
        Will return True if it is redundant and False otherwise (which includes the case that the node_id
        is not found in the Graph)

        :param node_id: ID of the node for which we are checking redundancy
        :type node_id: frozenset OR int
        :return: True if the node exists and is redundant (i.e. has more than 1 photon). False otherwise.
        :rtype: bool
        """
        if isinstance(node_id, int):
            node_id = frozenset([node_id])

        if node_id in self.node_dict.keys():
            return self.node_dict[node_id].count_redundancy() > 1
        else:
            warnings.warn("Node does not exist")
            return False

    def remove_id_from_redundancy(self, node_id, removal_id=None):
        """
        Remove a photon from the redundantly encoded node.
        If no photon id is specified, then the first photon in the node is removed.

        :param node_id: the id of the node from which we want to remove a photon
        :type node_id: int OR frozenset
        :param removal_id: id of the photon to remove inside the node (optional)
        :param removal_id: None OR int
        :return: nothing
        :rtype: None
        """
        if isinstance(node_id, int):
            node_id = frozenset([node_id])

        if node_id in self.node_dict.keys():
            node = self.get_node_by_id(node_id)
            if len(node_id) > 1:
                if removal_id is None:
                    node.remove_first_id()
                else:
                    node.remove_id(removal_id)
                new_node_id = node.get_id()
                self.node_dict[new_node_id] = self.node_dict.pop(node_id)
            else:
                assert node.count_redundancy() == 1  # just to make sure
                self.data.remove_node(self.node_dict[node_id])
                self.node_dict.pop(node_id)
        else:
            warnings.warn("No node is removed since node id does not exist.")

    def measure_x(self, node_id):
        """
        Measure a given node in the X basis.
        If the node contains a single photon, this measurement removes the node and disconnects all edges of this node.
        If the node is redundantly encoded, this measurement removes one photon from the node.

        :param node_id: the ID of the node to measure
        :type node_id: int OR frozenset
        :return: nothing
        :rtype: None
        """
        if isinstance(node_id, int):
            node_id = frozenset([node_id])

        if node_id in self.node_dict.keys():
            cnode = self.node_dict[node_id]
            if cnode.count_redundancy() == 1:
                self.data.remove_node(cnode)
                self.node_dict.pop(node_id, None)
            else:
                new_node_id = set(node_id)
                cnode.remove_id(new_node_id.pop())
                self.node_dict[frozenset(new_node_id)] = self.node_dict.pop(node_id)
        else:
            warnings.warn("No action is applied since node id does not exist.")

    def measure_y(self, node_id):
        """
        Apply a y measurement to the graph at the qubit corresponding
        to node_id

        :param node_id: the node id of the qubit which we are measuring
        :type node_id: int or frozenset
        :return: nothing
        :rtype: None
        """
        self.local_complementation(node_id)
        self.remove_node(
            node_id
        )  # TODO: double check, does redundant encoding matter here?

    def measure_z(self, node_id):
        self.remove_node(node_id)

    def draw(self, show=True, ax=None, with_labels=True):
        """
        Draw the underlying networkX graph

        :param show: if True, the Graph is shown. If False, the Graph is drawn but not displayed
        :type show: bool
        :param ax: axis on which to draw the plot (optional)
        :type ax: matplotlib.axis
        :param with_labels:
        :type with_labels:
        :return: nothing
        :rtype: None
        """
        draw_graph(self, show=show, ax=ax, with_labels=with_labels)

    @classmethod
    def valid_datatype(cls, data):
        return isinstance(
            data, (int, frozenset, nx.Graph, Iterable)
        ) and not isinstance(data, np.ndarray)

    def _node_as_qunode(self, node):
        """
        Converts a node_id (either int or frozenset) to its equivalent QuNode object in the graph OR
        returns the QuNode object unchanged

        :param node: either the node we want, or the node_id of the node
        :type node: int, frozenset, QuNode
        :return: the corresponding node in the graph
        :rtype: QuNode
        """
        if isinstance(node, int):
            node = frozenset([node])

        if isinstance(node, frozenset):
            node = self.node_dict[node]

        return node

    def local_complementation(self, node_id):
        """
        Takes the local complementation of the graph around node.

        Local complementation: let n(node) be the set of neighbours of node. If a, b in n(node) and (a, b) is in
        the set of edges E of graph, then remove (a, b) from E. If a, b in n(node) and (a, b) is NOT in E, then
        add (a, b) into E.

        :param node_id: the ID of the node around which local complementation should take place
        :type node_id: int or frozenset
        :return: nothing
        :rtype: None
        """
        # TODO: refactor to use local_comp_graph if determined to be more efficient
        # currently, we're not doing so because local_comp_graph handles ints rather than QuNode objects
        # Implementation 1
        # self.data = local_comp_graph(self.data, node_id)

        # Implementation 2
        neighbors = self.get_neighbors(node_id)
        neighbor_pairs = itertools.combinations(neighbors, 2)
        for a, b in neighbor_pairs:
            if self.edge_exists(a, b):
                self.remove_edge(a, b)
            else:
                self.add_edge(a, b)

    def merge(self, id1, id2):
        """
        Merges two nodes in the graph by:

        1. Creating a new node, new_node, where the node_id is union(node1_id, node2_id) --> union of the frozen sets
        2. For any edges of the form (a, node1) or (a, node2) with a in V (the set of nodes of the graph),
           add the edge (a, new_node) to the graph
        3. Remove node1, node2 and all their associated edges

        Note that this is NOT the fusion gate function, as it doesn't take into account possibility of
        failure. We also DO NOT check whether there is an existing edge between node1, node2 prior to merging.
        It is merely a function that deterministically merges nodes.

        :param id1: the ID of the first node in the graph to merge
        :type id1: int or frozenset
        :param id2: the ID of the second node in the graph to merge
        :type id2: int or frozenset
        :return: nothing
        :rtype: None
        """
        if isinstance(id1, int):
            id1 = frozenset([id1])
        if isinstance(id2, int):
            id2 = frozenset([id2])

        new_id = frozenset(set(id1).union(set(id2)))

        node1 = self.node_dict[id1]
        node2 = self.node_dict[id2]

        self.add_node(new_id)
        for edge in list(self.data.edges(node1)) + list(self.data.edges(node2)):
            other_node = edge[0] if edge[1] is node1 else edge[1]
            self.add_edge(other_node, new_id)

        self.remove_node(id1)
        self.remove_node(id2)


class MixedGraph(StateRepresentationBase):
    """
    Mixed Graph representation

    TODO: finish the implementation

    """

    def __init__(self, data, *args, **kwargs):
        if isinstance(data, int):
            self._mixture = [
                (1.0, Graph(data)),
            ]
        elif isinstance(data, Graph):
            self._mixture = [
                (1.0, data),
            ]
        elif isinstance(data, list):
            assert all(
                isinstance(p_i, float) and isinstance(t_i, Graph) for (p_i, t_i) in data
            )
            assert all
            self._mixture = data
        else:
            raise TypeError(
                f"Cannot initialize the stabilizer representation with datatype: {type(data)}"
            )

    @classmethod
    def valid_datatype(cls, data):
        valid = isinstance(data, (int, Graph, list))
        if isinstance(data, list):
            valid = valid and all(
                isinstance(p_i, float) and isinstance(t_i, Graph) for (p_i, t_i) in data
            )
        return valid

    @property
    def n_qubits(self):
        """
        Returns the number of qubits in the stabilizer state

        :return: the number of qubits in the state
        :rtype: int
        """
        return self._mixture[0][1].n_qubits
