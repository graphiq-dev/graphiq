r"""
State representations:
1. Graph representation
2. Density matrix
3. Stabilizer
"""
import networkx as nx
import numpy as np

import src.backends.density_matrix_functions as dmf
import src.backends.graph_functions as gf

class StateRepresentation:
    """
    Base class for state representation
    """
    def __init__(self,state_data):
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
    we may need to keep track of local unitaries that convert the state to the graph state represented by the graph
    """
    def __init__(self,state_data):
        # to be implemented later
        raise  NotImplementedError('')


class DensityMatrix(StateRepresentation):
    """
    Density matrix of a graph state
    """
    def __init__(self,state_data):
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
    def __init__(self,state_data):
        # to be implemented
        raise NotImplementedError('')
    def get_rep(self):
        raise NotImplementedError('')




# helper functions for density matrix related calculation below
