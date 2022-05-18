r"""
State representations:
1. Graph representation
2. Density matrix
3. Stabilizer
"""
import networkx as nx
import numpy as np
import warnings

from src.backends.graph_functions import QuNode

import src.backends.density_matrix_functions as dmf
import src.backends.graph_functions as gf


class StateRepresentation:
    """
    Base class for state representation
    """
    def __init__(self,state_data,*args, **kwargs):
        """
        Construct an empty state representation
        :param state_data: some input data about the state
        """
        raise ValueError('Base class StateRepresentation is abstract: it does not support function calls')
    def get_rep(self):
        """
        Return the representation of the state
        """
        raise ValueError('Base class StateRepresentation is abstract: it does not support function calls')





class GraphRep(StateRepresentation):
    """
    Graph representation of a graph state.
    As the intermediate states of the process may not be graph states (but assuming still stabilizer states),
    we may need to keep track of local cliffcords that convert the state to the graph state represented by the graph
    """

    def __init__(self, state_data, initial_counter, root_node_id):
        self.initial_counter = initial_counter
        self.root_node, self.node_dict, self.rep = gf.convert_data_to_graph(root_node_id,initial_counter,state_data)
        self.local_cliffords = None

    def add_node(self, node_to_add):
        """
        Adds a node to the graph.
        It allows node_to_add to be one of following types:
            QuNode
            int
            frozenset
        """
        if isinstance(node_to_add, QuNode):
            node_id = node_to_add.get_id()
            node_to_add.set_id_with_prefix(self.initial_counter, node_id)
            if node_id not in self.node_dict.keys():
                self.node_dict[node_id] = node_to_add
                self.rep.add_node(node_to_add)
            else:
                warnings.warn('Node already in the graph. Check node identifier.')
        elif isinstance(node_to_add, int):
            # node_to_add is just a node id; create the node first if it does not exist
            if node_to_add not in self.node_dict.keys():
                tmp_node = QuNode(self.initial_counter + node_to_add)
                self.node_dict[node_to_add] = tmp_node
                self.graph.add_node(tmp_node)
            else:
                warnings.warn('Node already in the graph. Check node identifier.')
        elif isinstance(node_to_add, frozenset):
            if node_to_add not in self.node_dict.keys():
                tmp_node = QuNode(node_to_add)
                tmp_node.set_id_with_prefix(self.initial_counter, node_to_add)
                self.node_dict[node_to_add] = tmp_node
                self.rep.add_node(tmp_node)
            else:
                warnings.warn('Node already in the graph. Check node identifier.')
        else:
            # invalid data Type
            raise ValueError('Invalid data for the node to be added.')

    def add_edge(self,first_node,second_node):
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

        if isinstance(node_id1,int):
            node_index1 = node_id1 - self.initial_counter
        else:
            tmp_id_list = list(node_id1)
            tmp_id_list2 = [id - self.initial_counter for id in tmp_id_list]
            node_index1 = frozenset(tmp_id_list2)
        if isinstance(node_id1,int):
            node_index2 = node_id2 - self.initial_counter
        else:
            tmp_id_list = list(node_id2)
            tmp_id_list2 = [id - self.initial_counter for id in tmp_id_list]
            node_index2 = frozenset(tmp_id_list2)

        if node_index1 in self.node_dict.keys() and node_index2 in self.node_dict.keys():
            self.rep.add_edge(self.node_dict[node_index1],self.node_dict[node_index2])
        else:
            warnings.warn('At least one of nodes do not exist. Not adding an edge.')

    def update_node_id(self,old_node_id,new_node_id):
        """
        Update the id of a node.
        """
        # update node id for a single node
        if old_node_id not in self.node_dict.keys():
            raise KeyError('Node does not exist')
        if new_node_id in self.node_dict.keys():
            raise KeyError('New identifier already in use')
        self.node_dict[old_node_id].set_id_with_prefix(self.initial_counter,new_node_id)
        self.node_dict[new_node_id] = self.node_dict.pop(old_node_id)


    def reassign_all_node_id(self,mapping):
        # mapping uses the old id as key and new id as value
        if len(self.node_dict) !=len(mapping):
            raise ValueError('Not all nodes are included.')
        if set(self.node_dict.keys()) != set(mapping.keys()):
            raise ValueError('Wrong node ids are included.')
        new_node_dict = dict()
        for key in mapping.keys():
            self.node_dict[key].set_id(mapping[key])
            new_node_dict[mapping[key]] =  self.node_dict[key]
        self.node_dict = new_node_dict

    def reassign_all_node_integer_id(self,starting_id, new_state_id):
        counter = starting_id
        tmp_dict = dict()
        for node_id in self.node_dict.keys():
            if isinstance(node_id,frozenset):
                new_id = list()
                new_id_short = list()
                for id in node_id:
                    new_id.append(counter+new_state_id)
                    new_id_short.append(counter)
                    counter += 1
                new_id = frozenset(new_id)
                new_id_short = frozenset(new_id_short)
            else:
                new_id = counter + new_state_id
                new_id_short = counter
                counter +=1
            tmp_node = self.node_dict[node_id]
            tmp_node.set_id(new_id)
            tmp_dict[new_id_short] = tmp_node
        self.node_dict = tmp_dict



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

    def get_edges_simple_id_form(self):
        return [(e[0].get_id_wo_prefix(self.initial_counter),e[1].get_id_wo_prefix(self.initial_counter)) for e in self.graph.edges]

    def get_nodes_id_form(self):
        return [node.get_id() for node in self.rep.nodes]

    def get_nodes_simple_id_form(self):
        return list(self.node_dict.keys())

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

    def get_graph_simple_id_form(self):
        tmp_graph = nx.Graph(self.get_edges_simple_id_form())
        nodelist = self.get_node_simple_id_form()
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
        if node_id in self.node_dict.keys():
            self.rep.remove_node(self.node_dict[node_id])
        else:
            warnings.warn('No node is removed since node id does not exist.')

    def remove_id_from_redundancy(self, node_id, removal_id=None):
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

    def get_rep(self):
        """
        Return the representation of the state
        """
        return self.rep

    def draw(self,draw_ax):
        """
        It allows one to draw the underlying networkX graph with matplotlib library.
        """
        nx.draw(self.rep, ax=draw_ax, with_labels=True, font_weight='bold')


class DensityMatrix(StateRepresentation):
    """
    Density matrix of a graph state
    """
    def __init__(self,state_data,*args, **kwargs):
        """
        Construct a DensityMatrix object and calculate the density matrix from state_data
        :param state_data: density matrix or a networkx graph
        """
        if isinstance(state_data,np.matrix) or isinstance(state_data,np.ndarray):
            # state_data is a numpy matrix or numpy ndarray

            # check if state_data is positive semidefinite
            if not dmf.is_psd(state_data):
                raise ValueError('The input matrix is not a density matrix')
            if not np.equal(np.trace(state_data),1):
                state_data = state_data / np.trace(state_data)

            self.rep = np.matrix(state_data)

        elif isinstance(state_data,nx.Graph):
            # state_data is a networkx graph
            number_qubits = state_data.number_of_nodes()
            mapping = dict(zip(state_data.nodes(), range(0, number_qubits)))
            edge_list = list(state_data.edges)
            final_state = dmf.create_n_plus_state(number_qubits)

            for edge in edge_list:
                final_state = dmf.apply_CZ(final_state, mapping[edge[0]], mapping[edge[1]])
            self.rep = final_state
        else:
            raise ValueError('Input data type is not supported for DensityMatrix.')

    def get_rep(self):
        return self.rep

    def apply_unitary(self,unitary_gate):
        """
        Apply a unitary gate on the state.
        Assuming the dimensions match; Otherwise, raise ValueError
        """
        if self.rep.shape == unitary_gate.shape:
            self.rep = unitary_gate @ self.rep @ np.tranpose(np.conjugate(unitary_gate))
        else:
            raise ValueError('The density matrix of the state has a different size from the unitary gate to be applied.')



class Stabilizer(StateRepresentation):
    def __init__(self,state_data,*args, **kwargs):
        # to be implemented
        raise NotImplementedError('')
    def get_rep(self):
        raise NotImplementedError('')




# helper functions for density matrix related calculation below
