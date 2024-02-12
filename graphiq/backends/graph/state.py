"""
Graph representation of quantum state
"""

import itertools
import copy
import warnings

import networkx as nx
import numpy as np

import graphiq.circuit.ops as ops
from graphiq.backends.lc_equivalence_check import is_lc_equivalent
from graphiq.backends.state_base import StateRepresentationBase
from graphiq.visualizers.graph import draw_graph


def _compile_lc_gates(lc_gate):
    """
    Compile local Clifford gates from a list of strings

    :param lc_gate: a list of local Clifford gates
    :type lc_gate: list[str]
    :return: a one-qubit gate
    :rtype: ops.OneQubitOperationBase
    """
    op_list = []
    for op in lc_gate:
        op_list.append(ops.name_to_class_map(op))
    return ops.simplify_local_clifford(op_list)


class Graph(StateRepresentationBase):
    """
    Graph representation of a graph state.
    As the intermediate states of the process may not be graph states (but assuming still stabilizer states),
    we may need to keep track of local Clifford gates that
    convert the state to the graph state represented by the graph.
    """

    def __init__(self, data, clifford_dict=None, *args, **kwargs):
        """
        Create a Graph representation object

        :param data: data used to construct the representation
        :type data: networkX.Graph
        :param clifford_dict: a dictionary that stores local Clifford for each node
        :type clifford_dict: dict
        :return: nothing
        :rtype: None
        """

        super().__init__(data, *args, **kwargs)
        if isinstance(data, nx.Graph):
            self._data = data
            if clifford_dict is None:
                for node in self._data.nodes:
                    self._data.nodes[node]["LC"] = [ops.Identity]
            else:
                for node in self._data.nodes:
                    if node in clifford_dict.keys():

                        self._data.nodes[node]["LC"] = _compile_lc_gates(
                            clifford_dict[node]
                        )
                    else:
                        self._data.nodes[node]["LC"] = [ops.Identity]
        else:
            raise TypeError(
                f"Cannot initialize the graph representation with datatype: {type(data)}"
            )

    @classmethod
    def valid_datatype(cls, data):
        """
        Validate the data type of the input data

        :param data: input data
        :type data: any
        :return: whether the data type is allowed for this class
        :rtype: bool
        """
        return isinstance(data, nx.Graph)

    @property
    def n_qubits(self):
        return self.n_nodes

    def find_lc(self, node_id):
        """
        Find the local Clifford gates corresponding to a node

        :param node_id: the node index
        :type node_id: int
        :return: local Clifford gates
        :rtype: ops.OneQubitOperationBase
        """
        if self._data.has_node(node_id):
            return self._data.nodes[node_id]["LC"]
        else:
            raise ValueError(f"Node with node ID {node_id} does not exist.")

    def update_lc(self, node_id, lc_gate):
        """
        Find the local Clifford gates corresponding to a node

        :param node_id: the node index
        :type node_id: int
        :param lc_gate: a list of local Clifford gates
        :type lc_gate: list(str)
        :return: local Clifford gates
        :rtype: ops.OneQubitOperationBase
        """
        if not self._data.has_node(node_id):
            raise ValueError(f"Node with node ID {node_id} does not exist.")
        if lc_gate is None or len(lc_gate) == 0:
            self._data.nodes[node_id]["LC"] = [ops.Identity]
        else:
            self._data.nodes[node_id]["LC"] = _compile_lc_gates(lc_gate)

    def add_node(self, node_to_add, lc_gate=None):
        """
        Add a node to the graph.

        :param node_to_add: node id to add to the Graph representation
        :type node_to_add: int
        :param lc_gate: a list of local Clifford gates
        :type lc_gate: list(str)
        :raises ValueError: if node_to_add is of an invalid datatype
        :return: nothing
        :rtype: None
        """
        if self._data.has_node(node_to_add):
            return
        if lc_gate is None:
            gate_to_add = [ops.Identity]
        else:
            gate_to_add = _compile_lc_gates(lc_gate)
        self._data.add_node(node_to_add, LC=gate_to_add)

    def add_edge(self, first_node, second_node):
        """
        Add an edge between two nodes. If any of these two nodes does not exist, no edge is added.

        :param first_node: the first node on which to add an edge
        :type first_node: int
        :param second_node: the second node on which to add an edge
        :type second_node: int
        :return: nothing
        :rtype: None
        """
        if not self._data.has_node(first_node):
            self.add_node(first_node)
        if not self._data.has_node(second_node):
            self.add_node(second_node)
        self._data.add_edge(first_node, second_node)

    def get_edges(self):
        """
        Get all graph edges (entangled pairs) in the Graph representation

        :return: graph edges
        :rtype: list
        """
        return list(self.data.edges)

    def get_nodes(self):
        """
        Get all graph nodes (qubits) in the Graph representation

        :return: all nodes in the Graph
        :rtype: list
        """
        return list(self.data.nodes)

    def lc_equivalent(self, other_graph, mode="deterministic"):
        r"""
        Determines whether two graph states are local-Clifford equivalent or not, given the adjacency matrices of the two.
        It takes two adjacency matrices as input and returns a numpy.ndarray containing $n (2 \times 2 array)s$
        = clifford operations on each qubit.

        :param other_graph: the other graph against which to check LC equivalence
        :type other_graph: Graph
        :param mode: the chosen mode for finding solutions. It can be either 'deterministic' (default) or 'random'.
        :type mode: str
        :raises AssertionError: if the number of rows in the row reduced matrix is less than the rank of coefficient
            matrix or if the number of linearly dependent columns is not equal to $4n - rank$
            (for $n$ being the number of nodes in the graph)
        :return: If a solution is found, returns True and an array of single-qubit Clifford $2 \times 2$ matrices
            in the symplectic formalism. If not, graphs are not LC equivalent and returns False, None.
        :rtype: bool, numpy.ndarray or None
        """
        g1 = nx.to_numpy_array(self.data).astype(int)
        g2 = nx.to_numpy_array(other_graph.data).astype(int)
        return is_lc_equivalent(g1, g2, mode=mode)

    @property
    def n_nodes(self):
        """
        Returns the number of nodes in the Graph

        :return: the number of nodes in the Graph
        :rtype: int
        """
        return self._data.number_of_nodes()

    def get_neighbors(self, node_id):
        """
        Return the list of all neighbors (i.e. nodes connected by an edge) of the node with node_id

        :param node_id: the ID of the node which we want to find the neighbours of
        :type node_id: int
        :return: a list of neighbours for the node with node_id
        :rtype: list
        """
        if self._data.has_node(node_id):
            return list(self._data.neighbors(node_id))
        else:
            raise ValueError(f"Node with node ID {node_id} does not exist.")

    def draw(self, show=True, ax=None, with_labels=True):
        """
        Draw the underlying networkX graph

        :param show: if True, the Graph is shown. If False, the Graph is drawn but not displayed
        :type show: bool
        :param ax: axis on which to draw the plot (optional)
        :type ax: matplotlib.Axis
        :param with_labels:
        :type with_labels:
        :return: nothing
        :rtype: None
        """
        draw_graph(self, show=show, ax=ax, with_labels=with_labels)

    def local_complementation(self, node_id, copy=False):
        """
        Takes the local complementation of the graph on the node indexed by node_id.

        Local complementation: let n(node) be the set of neighbours of node. If a, b in n(node) and (a, b) is in
        the set of edges E of graph, then remove (a, b) from E. If a, b in n(node) and (a, b) is NOT in E, then
        add (a, b) into E.

        The current implementation does not consider local Clifford gates. It assumes graph states.
        TODO: deal with general stabilizer states

        :param node_id: the ID of the node around which local complementation should take place
        :type node_id: int
        :return: the graph after the local complementation
        :rtype: Graph
        """

        if not Graph.is_graph_state(self):
            raise NotImplementedError(
                "Local complementation is not "
                "implemented for non-graph stabilizer states."
            )
        if copy:
            output_graph = self.copy()
        else:
            output_graph = self
        neighbors = self.get_neighbors(node_id)
        neighbor_pairs = itertools.combinations(neighbors, 2)
        for a, b in neighbor_pairs:
            if output_graph.data.has_edge(a, b):
                output_graph.data.remove_edge(a, b)
            else:
                output_graph.data.add_edge(a, b)
        return output_graph

    def copy(self):
        """
        Create a copy of this object

        :return: a copy of this Graph object
        :rtype: Graph
        """
        return copy.deepcopy(self)

    @classmethod
    def is_graph_state(cls, graph):
        """

        :param graph: an instance of Graph
        :type graph: Graph
        :return: True if graph is a graph state; False if graph is a non-graph stabilizer state
        :rtype: bool
        """
        assert isinstance(graph, Graph)
        # Check if all local Clifford gates are Identity
        for node in graph.data.nodes:
            if graph.data.nodes[node]["LC"] != [ops.Identity]:
                return False
        return True


class MixedGraph(StateRepresentationBase):
    """
    A mixed state representation using the graph representation, where the mixture is represented as a list of
    pure states (graphs) and an associated mixture probability.
    """

    def __init__(self, data, *args, **kwargs):
        if isinstance(data, Graph):
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
                f"Cannot initialize the graph representation with datatype: {type(data)}"
            )

    @classmethod
    def valid_datatype(cls, data):
        valid = isinstance(data, (Graph, list))
        if isinstance(data, list):
            valid = valid and all(
                isinstance(p_i, float) and isinstance(t_i, Graph) for (p_i, t_i) in data
            )
        return valid
