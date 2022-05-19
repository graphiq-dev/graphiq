from functools import reduce

import numpy as np


def sigmax():
    """
    Return sigma X matrix
    :return: sigma X matrix
    :rtype: numpy.matrix
    """
    return np.array([[0, 1], [1, 0]])


def sigmay():
    """
    Return sigma Y matrix
    :return: sigma Y matrix
    :rtype: numpy.array
    """
    return np.array([[0, -1j], [1j, 0]])


def sigmaz():
    """
    Return sigma Z matrix
    :return: sigma Z matrix
    :rtype: numpy.array
    """
    return np.array([[1, 0], [0, -1]])


def hadamard():
    """
    Return the Hadamard matrix for a qubit
    :return: sigma X matrix
    :rtype: numpy.array
    """
    return np.array([[1, 1], [1, -1]]) / np.sqrt(2)


def ketx0_state():
    """
    Return normalized eigenvector of sigma x matrix with eigenvalue +1
    """
    return 1/np.sqrt(2) * np.array([[1], [1]])


def ketx1_state():
    """
    Return normalized eigenvector of sigma x matrix with eigenvalue -1
    """
    return 1/np.sqrt(2) * np.array([[1], [-1]])


def ketz0_state():
    """
    Return normalized eigenvector of sigma z matrix with eigenvalue +1
    """
    return np.array([[1], [0]])


def ketz1_state():
    """"
    Return normalized eigenvector of sigma z matrix with eigenvalue -1
    """
    return np.array([[0], [1]])


def kety0_state():
    """
    Return normalized eigenvector of sigma y matrix with eigenvalue +1
    """
    return 1/np.sqrt(2) * np.array([[1], [1j]])


def kety1_state():
    """
    Return normalized eigenvector of sigma y matrix with eigenvalue -1
    """
    return 1/np.sqrt(2) * np.array([[1], [-1j]])


def get_controlled_gate(number_qubits, control_qubit, target_qubit, target_gate):
    """
    Define a controlled unitary gate
    :params number_qubits: specify the number of qubits in the system
    :params control_qubit: specify the index of the control qubit (starting from zero)
    :params target_qubit: specify the index of the target qubit
    :params target_gate: specify the gate to be applied conditioned on the control_qubit in the ket one state
    :type number_qubits: int
    :type control_qubit: int
    :type target_qubit: int
    :type target_gate: numpy.array
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
    return gate_cond0 + gate_cond1


def get_single_qubit_gate(number_qubits, qubit_position, target_gate):
    """
    A helper function to obtain the resulting matrix after tensoring the necessary identities
    :param number_qubits: number of qubits in the system
    :param qubit_position: the position of qubit that the target_gate acts on, qubit index starting from zero
    :type number_qubits: int
    :type qubit_position: int
    :return: This function returns the resulting matrix that acts on the whole state
    """

    final_gate = np.kron(np.identity(2**qubit_position), target_gate)
    final_gate = np.kron(final_gate, np.identity(2**(number_qubits-qubit_position-1)))
    return final_gate


def swap_two_qubits(state_matrix, qubit1_position, qubit2_position):
    """
    Swap two qubits by three CNOT gates
    Assuming state_matrix is a valid density matrix
    """
    number_qubits = int(np.log2(np.sqrt(state_matrix.size)))
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
    number_qubits = int(np.log2(np.sqrt(state_matrix.size)))
    target_op1 = np.transpose(np.conjugate(ketz0_state()))
    target_op2 = np.transpose(np.conjugate(ketz1_state()))
    k0 = get_single_qubit_gate(number_qubits, qubit_position, target_op1)
    k1 = get_single_qubit_gate(number_qubits, qubit_position, target_op2)
    final_state = (k0 @ state_matrix @ np.transpose(np.conjugate(k0))
                   + k1 @ state_matrix @ np.transpose(np.conjugate(k1)))
    return final_state


def is_hermitian(input_matrix):
    """
    Check if a matrix is Hermitian
    :params input_matrix: an input matrix for checking Hermitianity
    :type input_matrix: numpy.array
    :return: True or False
    :rtype: bool
    """

    return np.array_equal(input_matrix, np.conjugate(input_matrix.T))


def is_psd(input_matrix):
    """
    Check if a matrix is positive semidefinite.
    This method works efficiently for positive definite matrix.
    For positive-semidefinite matrices, we add a small perturbation of the identity matrix.
    :params input_matrix: an input matrix for checking positive semidefiniteness
    :type input_matrix: numpy.ndarray
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
    Create a product state that consists n tensor factors of the ket plus state
    """
    final_state = np.array([[1]])
    rho_init = np.matmul(ketx0_state(),np.transpose(np.conjugate(ketx0_state())))
    for i in range(number_qubits):
        final_state = np.kron(final_state, rho_init)
    return final_state


def tensor(arr):
    return reduce(np.kron, arr)


def ket2dm(ket):
    return np.outer(ket, ket)


def partial_trace(rho, keep, dims, optimize=False):
    """Calculate the partial trace

    ρ_a = Tr_b(ρ)

    Parameters
    ----------
    ρ : 2D array
        Matrix to trace
    keep : array
        An array of indices of the spaces to keep after
        being traced. For instance, if the space is
        A x B x C x D and we want to trace out B and D,
        keep = [0,2]
    dims : array
        An array of the dimensions of each space.
        For instance, if the space is A x B x C x D,
        dims = [dim_A, dim_B, dim_C, dim_D]

    Returns
    -------
    ρ_a : 2D array
        Traced matrix
    """
    keep = np.asarray(keep)
    dims = np.asarray(dims)
    Ndim = dims.size
    Nkeep = np.prod(dims[keep])

    idx1 = [i for i in range(Ndim)]
    idx2 = [Ndim + i if i in keep else i for i in range(Ndim)]
    rho_a = rho.reshape(np.tile(dims, 2))
    rho_a = np.einsum(rho_a, idx1 + idx2, optimize=optimize)
    return rho_a.reshape(Nkeep, Nkeep)


def apply_CZ(state_matrix, control_qubit, target_qubit):
    """
    Compute the density matrix after applying the controlled-Z gate
    """
    number_qubits = int(np.log2(np.sqrt(state_matrix.size)))
    cz = get_controlled_gate(number_qubits, control_qubit, target_qubit, sigmaz())
    return cz @ state_matrix @ np.transpose(np.conjugate(cz))


