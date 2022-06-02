"""
Helper functions for the density matrix representation backend

Includes functions to generate commonly used matrices, apply certain gates, etc.

"""
from functools import reduce

import numpy as np
from scipy.linalg import sqrtm


def sigmax():
    """
    Return sigma X matrix

    :return: sigma X matrix
    :rtype: numpy.ndarray
    """
    return np.array([[0.0, 1.0], [1.0, 0.0]])


def sigmay():
    """
    Return sigma Y matrix

    :return: sigma Y matrix
    :rtype: numpy.ndarray
    """
    return np.array([[0.0, -1.0j], [1.0j, 0.0]])


def sigmaz():
    """
    Return sigma Z matrix

    :return: sigma Z matrix
    :rtype: numpy.ndarray
    """
    return np.array([[1.0, 0.0], [0.0, -1.0]])


def hadamard():
    """
    Return the Hadamard matrix for a qubit
    :return: Hadamard matrix
    :rtype: numpy.ndarray
    """
    return np.array([[1.0, 1.0], [1.0, -1.0]]) / np.sqrt(2)


def ketx0_state():
    """
    Return normalized eigenvector of sigma x matrix with eigenvalue +1

    :return: normalized eigenvector of sigma x matrix, with eigenvalue +1
    :rtype: numpy.ndarray
    """
    return 1/np.sqrt(2) * np.array([[1.0], [1.0]])


def ketx1_state():
    """
    Return normalized eigenvector of sigma x matrix with eigenvalue -1

    :return: normalized eigenvector of sigma x matrix, with eigenvalue -1
    :rtype: numpy.ndarray
    """
    return 1/np.sqrt(2) * np.array([[1.0], [-1.0]])


def ketz0_state():
    """
    Return normalized eigenvector of sigma z matrix with eigenvalue +1

    :return: normalized eigenvector of sigma z matrix, with eigenvalue +1
    :rtype: numpy.ndarray
    """
    return np.array([[1.0], [0.0]])


def ketz1_state():
    """"
    Return normalized eigenvector of sigma z matrix with eigenvalue -1

    :return: normalized eigenvector of sigma z matrix, with eigenvalue -1
    :rtype: numpy.ndarray
    """
    return np.array([[0.0], [1.0]])


def kety0_state():
    """
    Return normalized eigenvector of sigma y matrix with eigenvalue +1

    :return: normalized eigenvector of sigma y matrix, with eigenvalue +1
    :rtype: numpy.ndarray
    """
    return 1/np.sqrt(2) * np.array([[1.0], [1.0j]])


def kety1_state():
    """
    Return normalized eigenvector of sigma y matrix with eigenvalue -1

    :return: normalized eigenvector of sigma y matrix, with eigenvalue -1
    :rtype: numpy.ndarray
    """
    return 1/np.sqrt(2) * np.array([[1.0], [-1.0j]])


def projectorz0():
    """
    Returns the projector in the 0 computational basis for a single qubit

    :return: 0 computational basis projector for a single qubit
    :rtype: numpy.ndarray
    """
    return np.array([[1.0, 0.0], [0.0, 0.0]])


def projectorz1():
    """
    Returns the projector in the 1 computational basis for a single qubit

    :return: 1 computational basis projector for a single qubit
    :rtype: numpy.ndarray
    """
    return np.array([[0.0, 0.0], [0.0, 1.0]])


def get_controlled_gate(number_qubits, control_qubit, target_qubit, target_gate):
    """
    Define a controlled unitary gate
    :params number_qubits: specify the number of qubits in the system
    :type number_qubits: int
    :params control_qubit: specify the index of the control qubit (starting from zero)
    :type control_qubit: int
    :params target_qubit: specify the index of the target qubit
    :type target_qubit: int
    :params target_gate: specify the gate to be applied conditioned on the control_qubit in the ket one state
    :type target_gate: numpy.array

    :return: a controlled unitary gate on the appropriate qubits and with the appropriate target gate
    :rtype: numpy.ndarray
    """
    gate_cond0 = np.array([[1]])
    gate_cond1 = np.array([[1]])
    if control_qubit < target_qubit:

        # tensor identities before the control qubit
        gate_cond0 = np.kron(gate_cond0, np.identity(2**control_qubit))
        gate_cond1 = np.kron(gate_cond1, np.identity(2**control_qubit))

        # tensor the gate on the control qubit
        gate_cond0 = np.kron(gate_cond0, ketz0_state() @ np.transpose(np.conjugate(ketz0_state())))
        gate_cond1 = np.kron(gate_cond1, ketz1_state() @ np.transpose(np.conjugate(ketz1_state())))

        # the rest is identity for the gate action conditioned on zero
        gate_cond0 = np.kron(gate_cond0, np.identity(2**(number_qubits-control_qubit-1)))

        # tensor identities between the control qubit and the target qubit
        gate_cond1 = np.kron(gate_cond1, np.identity(2**(target_qubit-control_qubit-1)))
        # tensor the gate on the target qubit
        gate_cond1 = np.kron(gate_cond1, target_gate)
        # tensor identities after the target qubit
        gate_cond1 = np.kron(gate_cond1, np.identity(2**(number_qubits-target_qubit-1)))
    elif control_qubit > target_qubit:
        # tensor identities before the control qubit for the gate action conditioned on zero
        gate_cond0 = np.kron(gate_cond0, np.identity(2**control_qubit))

        # tensor identities before the target qubit
        gate_cond1 = np.kron(gate_cond1, np.identity(2**target_qubit))
        # tensor the gate on the target qubit
        gate_cond1 = np.kron(gate_cond1, target_gate)
        # tensor identities between the control qubit and the target qubit
        gate_cond1 = np.kron(gate_cond1, np.identity(2**(control_qubit-target_qubit-1)))

        # tensor the gate on the control qubit
        gate_cond0 = np.kron(gate_cond0, ketz0_state() @ np.transpose(np.conjugate(ketz0_state())))
        gate_cond1 = np.kron(gate_cond1, ketz1_state() @ np.transpose(np.conjugate(ketz1_state())))

        # tensor identities after the control qubit
        gate_cond0 = np.kron(gate_cond0, np.identity(2**(number_qubits-control_qubit-1)))
        gate_cond1 = np.kron(gate_cond1, np.identity(2**(number_qubits-control_qubit-1)))
    else:
        raise ValueError('Control qubit and target qubit cannot be the same qubit!')
    return gate_cond0 + gate_cond1


def get_single_qubit_gate(number_qubits, qubit_position, target_gate):
    """
    Returns the matrix resulting from the "target_gate" matrix, after it has been tensored with the necessary identities

    :param number_qubits: number of qubits in the system
    :type number_qubits: int
    :param qubit_position: the position of qubit that the target_gate acts on, qubit index starting from zero
    :type qubit_position: int
    :param target_gate: the single-qubit version of the target gate
    :type target_gate: numpy.ndarray
    :return: This function returns the resulting matrix that acts on the whole state
    :rtype: numpy.ndarray
    """

    final_gate = np.kron(np.identity(2**qubit_position), target_gate)
    final_gate = np.kron(final_gate, np.identity(2**(number_qubits-qubit_position-1)))
    return final_gate


def swap_two_qubits(state_matrix, qubit1_position, qubit2_position):
    """
    Swap two qubits by three CNOT gates. Assumes state_matrix is a valid density matrix

    :param state_matrix: Initial matrix, pre-swap
    :type state_matrix: numpy.ndarray
    :param qubit1_position: first qubit index
    :type qubit1_position: int
    :param qubit2_position: second qubit index
    :type qubit2_position: int

    :return: the matrix with swapped qubits
    :rtype: numpy.ndarray
    """
    number_qubits = int(np.log2(np.sqrt(state_matrix.size)))
    cnot12 = get_controlled_gate(number_qubits, qubit1_position, qubit2_position, sigmax())
    cnot21 = get_controlled_gate(number_qubits, qubit2_position, qubit1_position, sigmax())

    # SWAP gate can be decomposed as three CNOT gates
    swap = cnot12 @ cnot21 @ cnot12
    final_state = swap @ state_matrix @ np.transpose(np.conjugate(swap))
    return final_state


def trace_out_qubit(state_matrix, qubit_position):
    """
    Trace out the specified qubit from the density matrix. Assumes state_matrix is a valid density matrix

    :param state_matrix: initial matrix, pre-trace
    :type state_matrix: numpy.ndarray
    :param qubit_position: the position (index) of the qubit to trace out
    :type qubit_position: int
    :return: the density matrix of the state with the qubit traced out
    :rtype: numpy.ndarray
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
    Returns True if a matrix is Hermitian, False otherwise

    :params input_matrix: an input matrix for checking Hermitianity
    :type input_matrix: numpy.array
    :return: True or False
    :rtype: bool
    """
    return np.array_equal(input_matrix, np.conjugate(input_matrix.T))


def is_psd(input_matrix):
    """
    Check if a matrix is positive semidefinite. This method works efficiently for positive definite matrix.
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

    :param number_qubits: size (number of qubits) in the state to build
    :type number_qubits: int
    :return: the product state
    :rtype: numpy.ndarray
    """
    final_state = np.array([[1]])
    rho_init = np.matmul(ketx0_state(), np.transpose(np.conjugate(ketx0_state())))
    for i in range(number_qubits):
        final_state = np.kron(final_state, rho_init)
    return final_state


def tensor(arr):
    """
    Takes the kronecker product of each ndarray in arr

    :param arr: list of ndarrays which can be combined via a kronecker product
    :type arr: list
    :return: the kronecker product of all the ndarrays in arr
    :rtype: numpy.ndarray
    """
    return reduce(np.kron, arr)


def ket2dm(ket):
    """
    Turns a ket into a density matrix

    :param ket: the ket to transform into a density matrix
    :type ket: numpy.ndarray
    :return: the density matrix equivalent of ket
    :rtype: numpy.ndarray
    """
    return ket @ np.transpose(np.conjugate(ket))


def partial_trace(rho, keep, dims, optimize=False):
    """
    Calculates the partial trace
    ρ_a = Tr_b(ρ)

    :param rho: the (2D) matrix to trace
    :type rho: numpy.ndarray
    :param keep:  An array of indices of the spaces to keep after being traced. For instance, if the space is
                  A x B x C x D and we want to trace out B and D, keep = [0,2]
    :type keep: list OR numpy.ndarray
    :param dims: An array of the dimensions of each space. For instance, if the space is A x B x C x D,
                 dims = [dim_A, dim_B, dim_C, dim_D]
    :type dims: list OR numpy.ndarray
    :param optimize: parameter about how to treat Einstein Summation convention on the operands
    :type optimize: bool OR str
    :return: the traced (2D) matrix
    :rtype: numpy.ndarray
    """
    keep = np.asarray(keep)
    dims = np.asarray(dims)
    ndim = dims.size
    nkeep = np.prod(dims[keep])

    idx1 = [i for i in range(ndim)]
    idx2 = [ndim + i if i in keep else i for i in range(ndim)]
    rho_a = rho.reshape(np.tile(dims, 2))
    rho_a = np.einsum(rho_a, idx1 + idx2, optimize=optimize)
    return rho_a.reshape(nkeep, nkeep)


def apply_cz(state_matrix, control_qubit, target_qubit):
    """
    Compute the density matrix after applying the controlled-Z gate

    :param state_matrix: original density matrix (before application of CZ)
    :type state_matrix: numpy.ndarray
    :param control_qubit: index of the control qubit in state_matrix
    :type control_qubit: int
    :param target_qubit: index of the target qubit in state_matrix
    :type target_qubit: int
    :return: the density matrix output after applying controlled-Z gate
    :rtype: numpy.ndarray
    """
    number_qubits = int(np.log2(np.sqrt(state_matrix.size)))
    cz = get_controlled_gate(number_qubits, control_qubit, target_qubit, sigmaz())

    return cz @ state_matrix @ np.transpose(np.conjugate(cz))


def projectors_zbasis(number_qubits, measure_register):
    """
    Get the z projector basis for a state of number_qubits size, where the "measure_register" qubit
    is the only one projected

    :param number_qubits: total number of qubits in the state
    :type number_qubits: int
    :param measure_register: the index of the register to measure
    :type measure_register: int
    :return: a list of projectors (0 projector and 1 projector)
    :rtype: list (of 2 ndarrays)
    """
    if not (0 <= measure_register < number_qubits):
        raise ValueError("Register index must be at least 0 and less than the number of qubit registers")
    m0 = reduce(np.kron, [projectorz0() if i == measure_register else np.identity(2) for i in range(number_qubits)])
    m1 = reduce(np.kron, [projectorz1() if i == measure_register else np.identity(2) for i in range(number_qubits)])

    return [m0, m1]


def fidelity(rho, sigma):
    """
    Return the fidelity between states rho, sigma

    :param rho: the first state
    :type rho: numpy.ndarray
    :param sigma: the second state
    :type sigma: numpy.ndarray
    :return: the fidelity between 0 and 1
    :rtype: int
    """
    return np.real(np.trace(sqrtm(sqrtm(rho) @ sigma @ sqrtm(rho))) ** 2)


if __name__ == "__main__":
    n_qubit = 3
    register = 0
    # state = ketx0_state()
    # state = reduce(np.kron, n_qubit * [state @ np.conjugate(state.T)])

    state = (reduce(np.kron, n_qubit * [ketz0_state()]) + reduce(np.kron, n_qubit * [ketz1_state()])) / np.sqrt(2)
    state = state @ np.conjugate(state.T)

    p0, p1 = projectors_zbasis(n_qubit, register)

    # print(p0)
    # print(p1)
    print(state)
    res = np.trace(state @ p0)
    print(res)
    res = np.trace(state @ p1)
    print(res)

    out = p0 @ state @ p0 + p1 @ state @ p1
    print(out)

    # print(partial_trace(state, keep=list(set(range(n_qubit)) - register, dims=n_qubit * [2])))

    projectors = projectors_zbasis(n_qubit, register)
    out = sum([m @ state @ m for m in projectors])
    print(out)
