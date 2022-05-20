r"""
State representations:
1. Graph representation
2. Density matrix
3. Stabilizer
"""
from abc import ABC
import networkx as nx
import numpy as np
import warnings

from src.backends.density_matrix.functions import apply_CZ, create_n_plus_state, is_psd
from src.backends.graph.functions import QuNode

import src.backends.graph.functions as gf


class StateRepresentationBase(ABC):
    """
    Base class for state representation
    """
    def __init__(self, state_data, state_id, *args, **kwargs):
        """
        Construct an empty state representation
        :param state_data: some input data about the state
        """
        self.state_data = state_data
        self.state_id = state_id
        self.rep = None

    def __str__(self):
        return f"{self.__class__.__name__}\n{self.rep}"

    def get_rep(self):
        """
        Return the representation of the state
        """
        return self.rep


class GraphRep(StateRepresentationBase):
    """
    Graph representation of a graph state.
    As the intermediate states of the process may not be graph states (but assuming still stabilizer states),
    we may need to keep track of local cliffcords that convert the state to the graph state represented by the graph
    """

    def __init__(self, state_data, root_node_id, *args, **kwargs):
        """
        :params state_data: data used to construct the representation
        :params state_id: a unique identifier for the state
        :params root_node_id: a node id for the root node
        """

        self.root_node, self.node_dict, self.rep = gf.convert_data_to_graph(state_data,root_node_id)
        self.local_cliffords = None # set this later

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
                self.rep.add_node(node_to_add)
            else:
                warnings.warn('Node already in the graph. Check node identifier.')
        elif isinstance(node_to_add, int) or isinstance(node_to_add, frozenset):
            # node_to_add is just a node id; create the node first if it does not exist
            if node_to_add not in self.node_dict.keys():
                tmp_node = QuNode(node_to_add)
                self.node_dict[node_to_add] = tmp_node
                self.rep.add_node(tmp_node)
            else:
                warnings.warn('Node already in the graph. Check node identifier.')
        else:
            # invalid data Type
            raise ValueError('Invalid data for the node to be added.')

    def add_edge(self,first_node,second_node):
        """
        Add an edge between two nodes. If any of these two nodes does not exist, no edge is added.
        """
        if isinstance(first_node,QuNode):
            node_id1 = first_node.get_id()
        elif isinstance(first_node,int) or isinstance(first_node,frozenset):
            node_id1 = first_node
        else:
            raise ValueError('Not supporting input data type')
        if isinstance(second_node,QuNode):
            node_id2 = second_node.get_id()
        elif isinstance(second_node, int) or isinstance(second_node, frozenset):
            node_id2 = second_node
        else:
            raise ValueError('Not supporting input data type')


        if node_id1 in self.node_dict.keys() and node_id2 in self.node_dict.keys():
            self.rep.add_edge(self.node_dict[node_id1],self.node_dict[node_id2])
        else:
            warnings.warn('At least one of nodes do not exist. Not adding an edge.')





    def get_edges(self):
        return [e for e in self.rep.edges]

    def get_nodes(self):
        return list(self.node_dict.values())

    def get_node_by_id(self,node_id):
        if node_id in self.node_dict:
            return self.node_dict[node_id]
        else:
            return None

    def get_edges_id_form(self):
        return [(e[0].get_id(),e[1].get_id()) for e in self.rep.edges]


    def get_nodes_id_form(self):
        return [node.get_id() for node in self.rep.nodes]



    def get_root_node(self):
        return self.root_node

    def get_node_dict(self):
        return self.node_dict

    def n_node(self):
        return len(self.node_dict.keys())

    def n_qubit(self):
        number_qubit = 0
        for node_id in self.node_dict.keys():
            if isinstance(node_id,frozenset):
                number_qubit += len(frozenset)
            else:
                number_qubit += 1
        return number_qubit

    def n_redundant_encoding_node(self):
        number_redundant_node = 0
        for node_id in self.node_dict.keys():
            if isinstance(node_id,frozenset):
                number_redundant_node += 1
        return number_redundant_node

    def get_graph_id_form(self):
        tmp_graph = nx.Graph(self.get_edges_id_form())
        nodelist = self.get_node_id_form()
        if set(nodelist) != set(tmp_graph.nodes()):
            for node in nodelist:
                tmp_graph.add_node(node)
        return tmp_graph


    def get_neighbors(self,node_id):
        """
        Return the list of all neighbors of the node with node_id
        """
        neighbor_list = list()
        if node_id in self.node_dict.keys():
            cnode = self.node_dict[node_id]
            all_nodes = self.node_dict.values()

            for node in all_nodes:
                if (node,cnode) in self.rep.edges() or (cnode,node) in self.rep.edges():
                    neighbor_list.append(node)
        return neighbor_list

    def remove_node(self, node_id):
        """
        Remove a node from the graph and remove all edges of the node
        """
        if node_id in self.node_dict.keys():
            self.rep.remove_node(self.node_dict[node_id])
        else:
            warnings.warn('No node is removed since node id does not exist.')

    def remove_id_from_redundancy(self, node_id, removal_id=None):
        """
        Remove a photon from the redunantly encoded node.
        If no photon id is specified, then the first photon in the node is removed.
        """
        if node_id in self.node_dict.keys():
            if isinstance(node_id,frozenset):
                node = self.get_node_by_id(node_id)
                if removal_id is None:
                    node.remove_first_id()
                else:
                    node.remove_id(removal_id)
                new_node_id = node.get_id()
                if node.count_redundancy() == 0:
                    self.rep.remove_node(self.node_dict[node_id])
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
                self.rep.remove_node(cnode)
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

    def draw(self, draw_ax):
        """
        It allows one to draw the underlying networkX graph with matplotlib library.
        """
        nx.draw(self.rep, ax=draw_ax, with_labels=True, font_weight='bold')


class DensityMatrix(StateRepresentationBase):
    """
    Density matrix of a graph state
    """
    def __init__(self, state_data, state_id, *args, **kwargs):
        """
        Construct a DensityMatrix object and calculate the density matrix from state_data
        :param state_data: density matrix or a networkx graph
        """
        super().__init__(state_data, state_id, *args, **kwargs)

        self.state_id = state_id
        if isinstance(state_data, np.ndarray):
            # state_data is a numpy ndarray

            # check if state_data is positive semi-definite
            if not is_psd(state_data):
                raise ValueError('The input matrix is not a density matrix')
            if not np.equal(np.trace(state_data), 1):
                state_data = state_data / np.trace(state_data)

            self.rep = state_data

        elif isinstance(state_data, nx.Graph) or isinstance(state_data,GraphRep):
            # state_data is a networkx graph
            if isinstance(state_data,GraphRep):
                graph_data = state_data.get_rep()
            else:
                graph_data = state_data
            number_qubits = state_data.number_of_nodes()
            mapping = dict(zip(state_data.nodes(), range(0, number_qubits)))
            edge_list = list(state_data.edges)
            final_state = create_n_plus_state(number_qubits)

            for edge in edge_list:
                final_state = apply_CZ(final_state, mapping[edge[0]], mapping[edge[1]])
            self.rep = final_state


        else:
            raise ValueError('Input data type is not supported for DensityMatrix.')



    def apply_unitary(self, unitary):
        """
        Apply a unitary on the state.
        Assuming the dimensions match; Otherwise, raise ValueError
        """
        if self.rep.shape == unitary.shape:
            self.rep = unitary @ self.rep @ np.transpose(np.conjugate(unitary))
        else:
            raise ValueError('The density matrix of the state has a different size from the unitary gate to be applied.')

    def apply_measurement(self, projectors):
        if self.rep.shape == projectors[0].shape:
            probs = [np.real(np.trace(self.rep @ m)) for m in projectors]

            outcome = np.random.choice([0, 1], p=probs/np.sum(probs))
            m, norm = projectors[outcome], probs[outcome]
            # TODO: this is the dm CONDITIONED on the measurement outcome
            # this assumes that the projector, m, has the properties: m = sqrt(m) and m = m.dag()
            self.rep = (m @ self.rep @ np.transpose(np.conjugate(m))) / norm

            # self.rep = sum([m @ self.rep @ m for m in projectors])  # TODO: this is the dm unconditioned on the outcome
        else:
            raise ValueError('The density matrix of the state has a different size from the POVM elements.')
        return outcome


class Stabilizer(StateRepresentationBase):
    def __init__(self, state_data, state_id, *args, **kwargs):
        # to be implemented
        raise NotImplementedError('')

    def get_rep(self):
        raise NotImplementedError('')
