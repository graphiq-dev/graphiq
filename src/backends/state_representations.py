r"""
State representations:
1. Graph representation
2. Density matrix
3. Stabilizer
"""
import networkx as nx
import numpy as np
import math


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
            if not is_psd(state_data):
                raise ValueError('The input matrix is not a density matrix')
            if not np.equal(np.trace(state_data),1):
                state_data = state_data / np.trace(state_data)

            self.rep = np.matrix(state_data)

        elif isinstance(state_data,nx.Graph):
            # state_data is a networkx graph
            number_qubits = state_data.number_of_nodes()
            mapping = dict(zip(state_data.nodes(), range(0, number_qubits)))
            edge_list = list(state_data.edges)
            final_state = create_n_plus_state(number_qubits)

            for edge in edge_list:
                final_state = apply_CZ(final_state, mapping[edge[0]], mapping[edge[1]])
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




def get_controlled_gate(number_qubits,control_qubit,target_qubit, target_gate):
    """
    Define a controlled unitary gate
    :params number_qubits: specify the number of qubits in the system
    :params control_qubit: specify the index of the control qubit (starting from zero)
    :params target_qubit: specify the index of the target qubit
    :params target_gate: specify the gate to be applied conditioned on the control_qubit in the ket one state
    :type number_qubits: int
    :type control_qubit: int
    :type target_qubit: int
    :type target_gate: numpy.matrix
    """
    gate_cond0 = np.array([[1]])
    gate_cond1 = np.array([[1]])
    if control_qubit < target_qubit:

        # tensor identities before the control qubit
        gate_cond0 = np.kron(gate_cond0,np.identity(2**control_qubit))
        gate_cond1 = np.kron(gate_cond1,np.identity(2**control_qubit))

        # tensor the gate on the control qubit
        gate_cond0 = np.kron(gate_cond0,ketz0_state() @ np.transpose(np.conjugate(ketz0_state())))
        gate_cond1 = np.kron(gate_cond1,ketz1_state() @ np.transpose(np.conjugate(ketz1_state())))

        # the rest is identity for the gate action conditioned on zero
        gate_cond0 = np.kron(gate_cond0,np.identity(2**(number_qubits-control_qubit-1)))

        # tensor identities between the control qubit and the target qubit
        gate_cond1 = np.kron(gate_cond1,np.identity(2**(target_qubit-control_qubit-1)))
        # tensor the gate on the target qubit
        gate_cond1 = np.kron(gate_cond1,target_gate)
        # tensor identities after the target qubit
        gate_cond1 = np.kron(gate_cond1,np.identity(2**(number_qubits-target_qubit-1)))
    elif control_qubit > target_qubit:
        # tensor identities before the control qubit for the gate action conditioned on zero
        gate_cond0 = np.kron(gate_cond0,np.identity(2**control_qubit))

        # tensor identities before the target qubit
        gate_cond1 = np.kron(gate_cond1,np.identity(2**target_qubit))
        # tensor the gate on the target qubit
        gate_cond1 = np.kron(gate_cond1,target_gate)
        # tensor identities between the control qubit and the target qubit
        gate_cond1 = np.kron(gate_cond1,np.identity(2**(control_qubit-target_qubit-1)))

        # tensor the gate on the control qubit
        gate_cond0 = np.kron(gate_cond0,ketz0_state() @ np.transpose(np.conjugate(ketz0_state())))
        gate_cond1 = np.kron(gate_cond1,ketz1_state() @ np.transpose(np.conjugate(ketz1_state())))

        # tensor identities after the control qubit
        gate_cond0 = np.kron(gate_cond0,np.identity(2**(number_qubits-control_qubit-1)))
        gate_cond1 = np.kron(gate_cond1,np.identity(2**(number_qubits-control_qubit-1)))
    else:
        raise ValueError('Control qubit and target qubit cannot be the same qubit!')
    return np.matrix(gate_cond0 + gate_cond1)

def get_single_qubit_gate(number_qubits,qubit_position,target_gate):
    """
    A helper function to obtain the resulting matrix after tensoring the necessary identities
    :param number_qubits: number of qubits in the system
    :param qubit_position: the position of qubit that the target_gate acts on, qubit index starting from zero
    :type number_qubits: int
    :type qubit_position: int
    :return: This function returns the resulting matrix that acts on the whole state
    """

    final_gate = np.kron(np.identity(2**qubit_position),target_gate)

    final_gate = np.matrix(np.kron(final_gate, np.identity(2**(number_qubits-qubit_position-1))))
    return final_gate


def swap_two_qubits(state_matrix, qubit1_position, qubit2_position):
    """
    Swap two qubits by three CNOT gates
    Assuming state_matrix is a valid density matrix
    """
    number_qubits = int(math.log2(math.sqrt(state_matrix.size)))
    cnot12 = get_controlled_gate(number_qubits,qubit1_position, qubit2_position, sigmax())
    cnot21 = get_controlled_gate(number_qubits,qubit2_position, qubit1_position, sigmax())

    # SWAP gate can be decomposed as three CNOT gates
    swap = cnot12 @ cnot21 @ cnot12
    final_state = swap @ state_matrix @ np.transpose(np.conjugate(swap))
    return final_state

def trace_out_qubit(state_matrix, qubit_position):
    """
    Trace out the specified qubit from the density matrix.
    Assuming state_matrix is a valid density matrix
    """
    number_qubits = int(math.log2(math.sqrt(state_matrix.size)))
    target_op1 = np.transpose(np.conjugate(ketzero_state()))
    target_op2 = np.transpose(np.conjugate(ketone_state()))
    k0 = get_single_qubit_gate(number_qubits,qubit_position,target_op1)
    k1 = get_single_qubit_gate(number_qubits,qubit_position,target_op2)
    final_state = (k0 @ state_matrix @ np.transpose(np.conjugate(k0))
                  + k1 @ state_matrix @ np.transpose(np.conjugate(k1)))
    return final_state


def is_hermitian(input_matrix):
    """
    Check if a matrix is Hermitian
    :params input_matrix: an input matrix for checking Hermitianity
    :type input_matrix: numpy.matrix
    :return: True or False
    :rtype: bool
    """
    return np.array_equal(input_matrix, np.conjugate(input_matrix).T)



def is_psd(input_matrix):
    """
    Check if a matrix is positive semidefinite.
    This method works efficiently for positive definite matrix.
    For positive-semidefinite matrices, we add a small perturbation of the identity matrix.
    :params input_matrix: an input matrix for checking positive semidefiniteness
    :type input_matrix: numpy.matrix
    :return: True or False
    :rtype: bool
    """
    perturbation = 1e-15
    # first check if it is a Hermitian matrix
    if not is_hermitian(input_matrix):
        return False
        
    perturbed_matrix = input_matrix + perturbation * np.identity(len(input_matrix))
    try:
        np.linalg.cholesky(perturbed_matrix)
        return True
    except np.linalg.LinAlgError:
        # np.linalg.cholesky throws this exception if the matrix is not positive definite
        return False



def create_n_plus_state(number_qubits):
    """
    Create a prudct state that consists n tensor factors of the ket plus state
    """
    final_state = np.array([[1]])
    ketPlus = 1/math.sqrt(2)*np.array([[1], [1]])
    rho_init = np.matrix(np.matmul(ketx0_state(),np.transpose(np.conjugate(ketx0_state()))))
    for i in range(number_qubits):
        final_state = np.kron(final_state, rho_init)
    return final_state

def apply_CZ(state_matrix,control_qubit,target_qubit):
    """
    Compute the density matrix after applying the controlled-Z gate
    """
    number_qubits = int(math.log2(math.sqrt(state_matrix.size)))
    cz = get_controlled_gate(number_qubits,control_qubit,target_qubit, sigmaz())
    return cz @ state_matrix @ np.transpose(np.conjugate(cz))

def sigmax():
    """
    Return sigma X matrix
    :return: sigma X matrix
    :rtype: numpy.matrix
    """
    return  np.matrix([[0,1],[1,0]])


def sigmay():
    """
    Return sigma Y matrix
    :return: sigma Y matrix
    :rtype: numpy.matrix
    """
    return  np.matrix([[0,-1j],[1j,0]])

def sigmaz():
    """
    Return sigma Z matrix
    :return: sigma Z matrix
    :rtype: numpy.matrix
    """
    return  np.matrix([[1,0],[0,-1]])

def ketx0_state():
    return 1/math.sqrt(2)*np.array([[1], [1]])

def ketx1_state():
    return 1/math.sqrt(2)*np.array([[1], [-1]])

def ketz0_state():
    return np.array([[1], [0]])

def ketz1_state():
    return np.array([[0], [1]])

def kety0_state():
    return 1/math.sqrt(2)*np.array([[1],[1j]])

def kety1_state():
    return 1/math.sqrt(2)*np.array([[1],[-1j]])
