"""
Helper functions for the density matrix representation backend

Includes functions to generate commonly used matrices, apply certain gates, etc.

"""
from functools import reduce
import string

from src.backends.density_matrix import numpy as np
from src.backends.density_matrix import eigh


def sigmax():
    """
    Return :math:`\\sigma_x` matrix

    :return: sigma X matrix
    :rtype: numpy.ndarray
    """
    return np.array([[0.0, 1.0], [1.0, 0.0]])


def sigmay():
    """
    Return :math:`\\sigma_y` matrix

    :return: sigma Y matrix
    :rtype: numpy.ndarray
    """
    return np.array([[0.0, -1.0j], [1.0j, 0.0]])


def sigmaz():
    """
    Return :math:`\\sigma_z`matrix

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


def phase():
    r"""
    Return the phase matrix :math:`P = \\begin{bmatrix} 1 & 0 \\\ 0 & i \\end{bmatrix}`

    :return: the phase matrix
    :rtype: numpy.ndarray
    """
    return np.diag(np.array([1.0, 1.0j]))


def state_ketx0():
    """
    Return normalized eigenvector of :math:`\\sigma_x` matrix with eigenvalue +1

    :return: normalized eigenvector of :math:`\\sigma_x` matrix, with eigenvalue +1
    :rtype: numpy.ndarray
    """
    return 1 / np.sqrt(2) * np.array([[1.0], [1.0]])


def state_ketx1():
    """
    Return normalized eigenvector of :math:`\\sigma_x` matrix with eigenvalue -1

    :return: normalized eigenvector of :math:`\\sigma_x` matrix, with eigenvalue -1
    :rtype: numpy.ndarray
    """
    return 1 / np.sqrt(2) * np.array([[1.0], [-1.0]])


def state_ketz0():
    """
    Return normalized eigenvector of :math:`\\sigma_z` matrix with eigenvalue +1

    :return: normalized eigenvector of :math:`\\sigma_z` matrix, with eigenvalue +1
    :rtype: numpy.ndarray
    """
    return np.array([[1.0], [0.0]])


def state_ketz1():
    """
    Return normalized eigenvector of :math:`\\sigma_z` matrix with eigenvalue -1

    :return: normalized eigenvector of :math:`\\sigma_z` matrix, with eigenvalue -1
    :rtype: numpy.ndarray
    """
    return np.array([[0.0], [1.0]])


def state_kety0():
    """
    Return normalized eigenvector of sigma y matrix with eigenvalue +1

    :return: normalized eigenvector of sigma y matrix, with eigenvalue +1
    :rtype: numpy.ndarray
    """
    return 1 / np.sqrt(2) * np.array([[1.0], [1.0j]])


def state_kety1():
    """
    Return normalized eigenvector of sigma y matrix with eigenvalue -1

    :return: normalized eigenvector of sigma y matrix, with eigenvalue -1
    :rtype: numpy.ndarray
    """
    return 1 / np.sqrt(2) * np.array([[1.0], [-1.0j]])


def projector_ketz0():
    """
    Returns the projector in the 0 computational basis for a single qubit

    :return: 0 computational basis projector for a single qubit
    :rtype: numpy.ndarray
    """
    return np.array([[1.0, 0.0], [0.0, 0.0]])


def projector_ketz1():
    """
    Returns the projector in the 1 computational basis for a single qubit

    :return: 1 computational basis projector for a single qubit
    :rtype: numpy.ndarray
    """
    return np.array([[0.0, 0.0], [0.0, 1.0]])


def get_two_qubit_controlled_gate(n_qubits, control_qubit, target_qubit, target_gate):
    """
    Define a two-qubit controlled unitary gate.

    :param n_qubits: specify the number of qubits in the system
    :type n_qubits: int
    :param control_qubit: specify the index of the control qubit (starting from zero)
    :type control_qubit: int
    :param target_qubit: specify the index of the target qubit
    :type target_qubit: int
    :param target_gate: specify the gate to be applied conditioned on the control_qubit in the ket one state
    :type target_gate: numpy.ndarray
    :raises ValueError: if the target and control qubits are the same
    :raises AssertionError: if the number of qubits is at most 1
    :return: a controlled unitary gate on the appropriate qubits and with the appropriate target gate
    :rtype: numpy.ndarray
    """
    assert n_qubits > 1
    if control_qubit < target_qubit:
        final_gate = np.kron(
            np.kron(np.eye(2**control_qubit), np.eye(2) - sigmaz()),
            np.eye(2 ** (target_qubit - control_qubit - 1)),
        )
        final_gate = np.kron(
            np.kron(final_gate, target_gate - np.eye(2)),
            np.eye(2 ** (n_qubits - target_qubit - 1)),
        )

    elif control_qubit > target_qubit:
        final_gate = np.kron(
            np.kron(np.eye(2**target_qubit), target_gate - np.eye(2)),
            np.eye(2 ** (control_qubit - target_qubit - 1)),
        )
        final_gate = np.kron(
            np.kron(final_gate, np.eye(2) - sigmaz()),
            np.eye(2 ** (n_qubits - control_qubit - 1)),
        )
    else:
        raise ValueError("Control qubit and target qubit cannot be the same qubit!")
    final_gate = np.eye(2**n_qubits) + final_gate / 2
    return final_gate


def get_one_qubit_gate(n_qubits, qubit_position, target_gate):
    """
    Returns the matrix resulting from the "target_gate" matrix, after it has been tensored with the necessary identities

    :param n_qubits: number of qubits in the system
    :type n_qubits: int
    :param qubit_position: the position of qubit that the target_gate acts on, qubit index starting from zero
    :type qubit_position: int
    :param target_gate: the single-qubit version of the target gate
    :type target_gate: numpy.ndarray
    :return: This function returns the resulting matrix that acts on the whole state
    :rtype: numpy.ndarray
    """
    if n_qubits == 1:
        return target_gate
    final_gate = np.kron(np.identity(2**qubit_position), target_gate)
    final_gate = np.kron(final_gate, np.identity(2 ** (n_qubits - qubit_position - 1)))
    return final_gate


def _get_multi_qubit_gate(n_qubits, target_gates_dict):
    """
    Returns the matrix resulting from the "target_gate" matrix, after tensoring with the necessary identities

    :param n_qubits: number of qubits in the system
    :type n_qubits: int
    :param target_gates_dict: a dictionary where the key is the qubit position
        and the value is a single-qubit gate that is non-identity components of the final gate
    :type target_gates_dct: dict
    :return: the resulting matrix that acts on the whole state
    :rtype: numpy.ndarray
    """
    qubit_positions = sorted(target_gates_dict)

    assert n_qubits >= len(qubit_positions)
    final_gate = 1
    previous_position = 0
    for position in qubit_positions:
        final_gate = np.kron(
            final_gate, np.identity(2 ** (position - previous_position))
        )
        final_gate = np.kron(final_gate, target_gates_dict[position])
        previous_position = position + 1
    if n_qubits - qubit_positions[-1] - 1 > 0:
        final_gate = np.kron(
            final_gate, np.identity(2 ** (n_qubits - qubit_positions[-1] - 1))
        )
    return final_gate


def get_multi_qubit_gate(n_qubits, qubit_positions, target_gates):
    """
    Returns the matrix resulting from the "target_gate" matrix, after it has been tensored with the necessary identities

    :param n_qubits: number of qubits in the system
    :type n_qubits: int
    :param qubit_positions: a list of positions for non-identity gates
    :type qubit_positions: list[int]
    :param target_gates: a list of gates
    :type target_gates: list[numpy.ndarray]
    :raises AssertionError: if the number of qubit positions to apply gates is not equal to the number of gates
    :return: the resulting matrix that acts on the whole state
    :rtype: numpy.ndarray
    """
    assert len(qubit_positions) == len(target_gates)
    target_gates_dict = {
        qubit_positions[i]: target_gates[i] for i in range(len(qubit_positions))
    }
    return _get_multi_qubit_gate(n_qubits, target_gates_dict)


def swap_two_qubits(state_matrix, qubit1_position, qubit2_position):
    """
    Swap two qubits by three CNOT gates. Assumes state_matrix is a valid density matrix.

    :param state_matrix: Initial matrix, pre-swap
    :type state_matrix: numpy.ndarray
    :param qubit1_position: first qubit index
    :type qubit1_position: int
    :param qubit2_position: second qubit index
    :type qubit2_position: int
    :raises AssertionError: if the number of qubits is at most 1
    :return: the matrix with swapped qubits
    :rtype: numpy.ndarray
    """
    n_qubits = int(np.log2(np.sqrt(state_matrix.size)))
    assert n_qubits > 1
    cnot12 = get_two_qubit_controlled_gate(
        n_qubits, qubit1_position, qubit2_position, sigmax()
    )
    cnot21 = get_two_qubit_controlled_gate(
        n_qubits, qubit2_position, qubit1_position, sigmax()
    )

    # SWAP gate can be decomposed as three CNOT gates
    swap = cnot12 @ cnot21 @ cnot12
    final_state = swap @ state_matrix @ np.transpose(np.conjugate(swap))
    return final_state


def get_reset_qubit_kraus(n_qubits, qubit_position):
    """
    Generate a list of Kraus operators for resetting one qubit among n-qubit state.

    :param n_qubits: number of qubits
    :type n_qubits: int
    :param qubit_position: the position of qubit to be reset
    :type qubit_position: int
    :return: Kraus operators corresponding to reset
    :rtype: [numpy.ndarray, numpy.ndarray]
    """
    kraus0 = np.array([[1, 0], [0, 0]])
    kraus1 = np.array([[0, 1], [0, 0]])
    full_kraus0 = get_one_qubit_gate(n_qubits, qubit_position, kraus0)
    # full_kraus1 is technically not a gate, but this function works
    full_kraus1 = get_one_qubit_gate(n_qubits, qubit_position, kraus1)
    return [full_kraus0, full_kraus1]


def trace_out_qubit(state_matrix, qubit_position):
    """
    Trace out the specified qubit from the density matrix. Assumes state_matrix is a valid density matrix.

    :param state_matrix: initial matrix, pre-trace
    :type state_matrix: numpy.ndarray
    :param qubit_position: the position (index) of the qubit to trace out
    :type qubit_position: int
    :return: the density matrix of the state with the qubit traced out
    :rtype: numpy.ndarray
    """
    n_qubits = int(np.log2(np.sqrt(state_matrix.size)))
    if n_qubits == 1:
        return np.array(np.trace(state_matrix))
    target_op1 = np.transpose(np.conjugate(state_ketz0()))
    target_op2 = np.transpose(np.conjugate(state_ketz1()))
    kraus0 = get_one_qubit_gate(n_qubits, qubit_position, target_op1)
    kraus1 = get_one_qubit_gate(n_qubits, qubit_position, target_op2)
    final_state = kraus0 @ state_matrix @ np.transpose(
        np.conjugate(kraus0)
    ) + kraus1 @ state_matrix @ np.transpose(np.conjugate(kraus1))
    return final_state


def is_hermitian(input_matrix):
    """
    Returns True if a matrix is Hermitian, False otherwise

    :param input_matrix: an input matrix for checking Hermitianity
    :type input_matrix: numpy.ndarray
    :return: True or False
    :rtype: bool
    """
    return np.allclose(input_matrix, np.conjugate(input_matrix.T))


def is_psd(input_matrix, perturbation=1e-15):
    """
    Check if a matrix is positive semidefinite. This method works efficiently for positive definite matrix.
    For positive-semidefinite matrices, we add a small perturbation of the identity matrix.

    :param input_matrix: an input matrix for checking positive semidefiniteness
    :type input_matrix: numpy.ndarray
    :param perturbation: small constant added to perturb the matrix
    :type perturbation: float
    :return: True or False
    :rtype: bool
    """
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


def is_density_matrix(input_matrix, perturbation=1e-15):
    """
    Check if the input_matrix is a valid density matrix, that is, positive semidefinite and unit trace.

    :param input_matrix: an input matrix to check
    :type input_matrix: numpy.ndarray
    :param perturbation: a small perturbation for checking positive semidefiniteness
    :type perturbation: float
    :return: True of input_matrix is a valid density matrix; False otherwise
    :rtype: bool
    """
    return is_psd(input_matrix, perturbation) and np.allclose(input_matrix.trace(), 1.0)


def is_pure(rho):
    """
    Determine if the input state rho is pure

    :param rho: a density matrix to check purity
    :type rho: numpy.ndarray
    :return: True if rho is pure; False if rho is mixed
    :rtype: bool
    """
    return np.allclose(np.real(np.trace(rho @ rho)), 1.0)


def create_n_product_state(n_qubits, qubit_state):
    """
    Create a product state that consists :math:`n` tensor factors of the qubit_state.

    :param n_qubits: size (number of qubits) in the state to build
    :type n_qubits: int
    :param qubit_state: the state of one qubit
    :type qubit_state: numpy.ndarray
    :raises ValueError: if the input qubit_state is neither a density matrix nor a ket vector
    :return: the product state of math:`n` tensor factors of qubit_state
    :rtype: numpy.ndarray
    """
    if qubit_state.shape[0] == qubit_state.shape[1]:
        return reduce(np.kron, n_qubits * [qubit_state])
    elif qubit_state.shape[1] == 1:
        return reduce(np.kron, n_qubits * [ket2dm(qubit_state)])
    else:
        raise ValueError(
            "The qubit_state should be either a density matrix of a ket vector"
        )


def create_n_plus_state(n_qubits):
    """
    Create a product state that consists :math:`n` tensor factors of :math:`|+\\rangle` state.

    :param n_qubits: size (number of qubits) in the state to build
    :type n_qubits: int
    :return: the product state of :math:`|+\\rangle`
    :rtype: numpy.ndarray
    """

    return create_n_product_state(n_qubits, state_ketx0())


def tensor(arr):
    """
    Takes the kronecker product of each ndarray in arr

    :param arr: list of ndarrays which can be combined via a kronecker product
    :type arr: list[numpy.ndarray]
    :return: the kronecker product of all the ndarrays in arr
    :rtype: numpy.ndarray
    """
    return reduce(np.kron, arr)


def ket2dm(ket):
    """
    Turns a ket into a density matrix.

    :param ket: the ket to transform into a density matrix
    :type ket: numpy.ndarray
    :return: the density matrix equivalent of ket
    :rtype: numpy.ndarray
    """
    return ket @ np.transpose(np.conjugate(ket))


def partial_trace(rho, keep, dims, optimize=False):
    """
    Calculates the partial trace
    :math:`\\rho_a = Tr_b(\\rho)`

    :param rho: the (2D) matrix to take the partial trace
    :type rho: numpy.ndarray
    :param keep:  An array of indices of the spaces to keep. For instance, if the space is
                :math:`A \\times B \\times C \\times D` and we want to trace out B and D, keep = [0,2]
    :type keep: list OR numpy.ndarray
    :param dims: An array of the dimensions of each space. For instance,
                if the space is :math:`A \\times B \\times C \\times D`,
                dims = [dim_A, dim_B, dim_C, dim_D]
    :type dims: list OR numpy.ndarray
    :param optimize: parameter about how to treat Einstein Summation convention on the operands
    :type optimize: bool OR str
    :return: the (2D) matrix after partial trace
    :rtype: numpy.ndarray
    """

    keep = np.asarray(keep)
    dims = np.asarray(dims)
    ndim = dims.size
    nkeep = np.prod(dims[keep])

    ssleft = "".join([string.ascii_lowercase[i] for i in range(ndim)]) + "".join(
        [string.ascii_uppercase[i] for i in range(ndim)]
    )
    ssright = "".join(
        [string.ascii_lowercase[i] for i in range(ndim) if i in keep]
    ) + "".join([string.ascii_uppercase[i] for i in range(ndim) if i in keep])
    superscript = ssleft + "->" + ssright

    rho_a = rho.reshape(np.tile(dims, 2))
    rho_a = np.einsum(superscript, rho_a)
    return rho_a.reshape(nkeep, nkeep)


def apply_cz(state_matrix, control_qubit, target_qubit):
    """
    Compute the density matrix after applying the controlled-Z gate.

    :param state_matrix: original density matrix (before application of CZ)
    :type state_matrix: numpy.ndarray
    :param control_qubit: index of the control qubit in state_matrix
    :type control_qubit: int
    :param target_qubit: index of the target qubit in state_matrix
    :type target_qubit: int
    :return: the density matrix output after applying controlled-Z gate
    :rtype: numpy.ndarray
    """
    n_qubits = int(np.log2(np.sqrt(state_matrix.size)))
    cz = get_two_qubit_controlled_gate(n_qubits, control_qubit, target_qubit, sigmaz())

    return cz @ state_matrix @ np.transpose(np.conjugate(cz))


def projectors_zbasis(n_qubits, measure_register):
    """
    Get the z projector basis for a state of number_qubits size, where the "measure_register" qubit
    is the only one projected

    :param n_qubits: total number of qubits in the state
    :type n_qubits: int
    :param measure_register: the index of the register to measure
    :type measure_register: int
    :raises ValueError: if measure_register is not a valid register index
    :return: a list of projectors (0 projector and 1 projector)
    :rtype: [numpy.ndarray, numpy.ndarray]
    """
    if not (0 <= measure_register < n_qubits):
        raise ValueError(
            "Register index must be at least 0 and less than the number of qubit registers"
        )
    projector0 = reduce(
        np.kron,
        [
            projector_ketz0() if i == measure_register else np.identity(2)
            for i in range(n_qubits)
        ],
    )
    projector1 = reduce(
        np.kron,
        [
            projector_ketz1() if i == measure_register else np.identity(2)
            for i in range(n_qubits)
        ],
    )

    return [projector0, projector1]


def sqrtm_psd(input_matrix):
    """
    Return the matrix square root of a positive semidefinite matrix input_matrix

    :param input_matrix: a positive semidefinite matrix
    :type input_matrix: numpy.ndarray
    :raise AssertionError: if the input_matrix is not positive semidefinite
    :return: the matrix square root of a positive semidefinite matrix input_matrix
    :rtype: numpy.ndarray
    """
    assert is_psd(input_matrix)
    eig_vals, eig_vecs = eigh(input_matrix)
    eig_vals = np.maximum(eig_vals, 0)
    return (eig_vecs * np.sqrt(eig_vals)) @ eig_vecs.T


def hermitianize(input_matrix):
    """
    Return a Hermitian matrix based on the input_matrix

    :param input_matrix: a matrix
    :type input_matrix: numpy.ndarray
    :return: a Hermitian matrix
    :rtype: numpy.ndarray
    """
    return (input_matrix + np.conjugate(input_matrix.T)) / 2


def fidelity(rho, sigma):
    """
    Return the fidelity between states rho, sigma assuming both are valid density matrices

    :math:`F(\\rho, \\sigma):=Tr[\\sqrt{\\sqrt{\\rho} \\sigma \\sqrt{\\rho}}]^2`

    If either rho or sigma is pure, then it simplifies as
    :math:`F(\\rho, \\sigma):=Tr[\\rho \\sigma]`

    :param rho: the first state
    :type rho: numpy.ndarray
    :param sigma: the second state
    :type sigma: numpy.ndarray
    :raises AssertionError: if not both rho and sigma are density matrices
    :return: the fidelity between 0 and 1
    :rtype: float
    """

    assert is_density_matrix(rho)
    assert is_density_matrix(sigma)

    if is_pure(rho) or is_pure(sigma):
        # if either one is pure, use the simplified expression
        return np.maximum(np.minimum(np.real(np.trace(rho @ sigma)), 1.0), 0.0)
    else:
        # if both are mixed, use the definition
        sqrt_rho = sqrtm_psd(rho)
        rho_sigma = sqrt_rho @ sigma @ sqrt_rho
        #  enforce it to be Hermitian to avoid numerical error
        rho_sigma = hermitianize(rho_sigma)

        rho_final = sqrtm_psd(rho_sigma)
        return np.maximum(np.minimum(np.real(np.trace(rho_final)) ** 2, 1.0), 0.0)


def trace_distance(rho, sigma):
    """
    Return the trace distance between two states rho and sigma

    :param rho: the first state
    :type rho: numpy.ndarray
    :param sigma: the second state
    :type sigma: numpy.ndarray
    :return: the trace distance between 0 and 1
    :rtype: float
    """
    return np.real(np.linalg.norm(rho - sigma, "nuc") / 2)


def bipartite_partial_transpose(rho, dim1, dim2, subsys):
    """
    Return the partial transpose matrix of a bipartite density matrix rho

    :param rho: the density matrix before applying partial transpose
    :type rho: numpy.ndarray
    :param dim1: dimension of the first system
    :type dim1: int
    :param dim2: dimension of the second system
    :type dim2: int
    :param subsys: index of the subsystem to take transpose
    :type subsys: int

    :raises AssertionError: if the dimension of the subsystems and the matrix dimension do not match
    :raises ValueError: if the input matrix is not bipartite
    :return: the partial transpose matrix of rho
    :rtype: numpy.ndarray
    """
    assert int(np.sqrt(rho.size)) == dim1 * dim2

    if subsys == 0:
        mask = [1, 0]
    elif subsys == 1:
        mask = [0, 1]
    else:
        raise ValueError(
            "The function bipartite_partial_transpose accepts only bipartite states."
        )
    pt_dims = np.arange(4).reshape(2, 2).T
    pt_index = np.concatenate(
        [
            [pt_dims[n, mask[n]] for n in range(2)],
            [pt_dims[n, 1 - mask[n]] for n in range(2)],
        ]
    )
    rho_pt = (
        rho.reshape(np.array([dim1, dim1, dim2, dim2]))
        .transpose(pt_index)
        .reshape(rho.shape)
    )
    return rho_pt


def negativity(rho, dim1, dim2):
    """
    Return the negativity of the matrix rho.

    :param rho: the density matrix to evaluate the negativity
    :type rho: numpy.ndarray
    :param dim1: dimension of the first system
    :type dim1: int
    :param dim2: dimension of the second system
    :type dim2: int
    :return: the negativity of rho
    :rtype: float
    """
    rho_pt = bipartite_partial_transpose(rho, dim1, dim2, 0)

    eig_vals, _ = np.linalg.eig(rho_pt)
    eig_vals = np.real(eig_vals)

    return np.sum(np.abs(eig_vals) - eig_vals) / 2


def project_to_z0_and_remove(rho, locations):
    """
    Return the density matrix after applying Z measurements on qubits specified by locations mask.
    It removes all these qubits under measurements.

    :param rho: the density matrix to evaluate the negativity
    :type rho: numpy.ndarray
    :param locations: a list of zeros/ones
    :type locations: list or numpy.ndarray
    :return: density matrix after measuring qubits specified by locations mask in Z basis
    :rtype: numpy.ndarray
    """
    n_qubits = len(locations)
    projector0 = reduce(
        np.kron,
        [
            projector_ketz0() if locations[i] else np.identity(2)
            for i in range(n_qubits)
        ],
    )
    new_rho = projector0 @ rho @ np.conjugate(projector0.T)
    new_rho = new_rho / np.trace(new_rho)

    keeps = []
    for i in range(n_qubits):
        if locations[i] == 0:
            keeps.append(i)
    dims = n_qubits * [2]
    final_rho = partial_trace(new_rho, keeps, dims)
    return final_rho


def parameterized_one_qubit_unitary(theta, phi, lam):
    r"""
    Define a generic 3-parameter one-qubit unitary gate.

    :math:`U(\\theta, \\phi, \\lambda) = \\begin{bmatrix} \\cos(\\frac{\\theta}{2}) & -e^{i \\lambda}
    \\sin(\\frac{\\theta}{2})\\\ e^{i \\phi}\\sin(\\frac{\\theta}{2}) &
    e^{i (\\phi+\\lambda)}\\cos(\\frac{\\theta}{2})\\end{bmatrix}`

    :param theta: an angle
    :type theta: float or double
    :param phi: an angle
    :type phi: float or double
    :param lam: an angle
    :type lam: float or double
    :return: a one-qubit unitary matrix
    :rtype: numpy.ndarray
    """
    gate = np.array(
        [
            [np.cos(theta / 2), -np.exp(1j * lam) * np.sin(theta / 2)],
            [
                np.exp(1j * phi) * np.sin(theta / 2),
                np.exp(1j * (phi + lam)) * np.cos(theta / 2),
            ],
        ]
    )
    return gate


def full_one_qubit_unitary(n_qubits, qubit_position, theta, phi, lam):
    r"""
    Define a generic 3-parameter one-qubit unitary gate that acts on the whole space.

    :math:`U(\\theta, \\phi, \\lambda) = \\begin{bmatrix} \\cos(\\frac{\\theta}{2}) & -e^{i \\lambda}
    \\sin(\\frac{\\theta}{2})\\\ e^{i \\phi}\\sin(\\frac{\\theta}{2}) &
    e^{i (\\phi+\\lambda)}\\cos(\\frac{\\theta}{2})\\end{bmatrix}`

    :param n_qubits: number of qubits
    :type n_qubits: int
    :param qubit_position: position of the target qubit
    :type qubit_position: int
    :param theta: an angle
    :type theta: float or double
    :param phi: an angle
    :type phi: float or double
    :param lam: an angle
    :type lam: float or double
    :return: a one-qubit unitary matrix that can be directly applied to an :math:`n`-qubit state
    :rtype: numpy.ndarray
    """
    gate = parameterized_one_qubit_unitary(theta, phi, lam)
    return get_one_qubit_gate(n_qubits, qubit_position, gate)


def full_two_qubit_controlled_unitary(
    n_qubits, ctr_qubit, target_qubit, theta, phi, lam, gamma
):
    """
    Define a generic 4-parameter two-qubit gate that is a controlled unitary gate.
    :math:`|0\\rangle \\langle 0|\\otimes I +
    e^{i \\gamma} |1\\rangle \\langle 1| \\otimes U(\\theta, \\phi, \\lambda)`,
    where :math:`U(\\theta,\\phi, \\lambda) =
    \\begin{bmatrix} \\cos(\\frac{\\theta}{2}) & -e^{i \\lambda} \\sin(\\frac{\\theta}{2}) \\\
    e^{i \\phi}\\sin(\\frac{\\theta}{2}) & e^{i (\\phi+\\lambda)}\\cos(\\frac{\\theta}{2})\\end{bmatrix}`

    :param n_qubits: number of qubits
    :type n_qubits: int
    :param ctr_qubit: position of the control qubit
    :type ctr_qubit: int
    :param target_qubit: position of the target qubit
    :type target_qubit: int
    :param theta: an angle
    :type theta: float or double
    :param phi: an angle
    :type phi: float or double
    :param lam: an angle
    :type lam: float or double
    :param gamma: a phase
    :type gamma: float or double
    :return: a two-qubit controlled unitary matrix that can be directly applied to an n-qubit state
    :rtype: numpy.ndarray
    """
    gate = np.exp(1j * gamma) * parameterized_one_qubit_unitary(theta, phi, lam)
    return get_two_qubit_controlled_gate(n_qubits, ctr_qubit, target_qubit, gate)


def is_unitary(input_matrix):
    """
    Check if the input matrix is a unitary matrix

    :param input_matrix: an input matrix
    :type input_matrix: numpy.ndarray
    :return: True if the input matrix is unitary; False otherwise
    :rtype: bool
    """
    if input_matrix.shape[0] != input_matrix.shape[1]:
        return False
    identity = np.eye(input_matrix.shape[0])
    adjoint = np.conjugate(input_matrix.T)
    return np.allclose(input_matrix @ adjoint, identity) and np.allclose(
        adjoint @ input_matrix, identity
    )


def check_equivalent_unitaries(unitary_op1, unitary_op2):
    """
    Check if two input matrices are equivalent unitaries up to a global phase

    :param unitary_op1: the first input matrix
    :type unitary_op1: numpy.ndarray
    :param unitary_op2: the second input matrix
    :type unitary_op2: numpy.ndarray
    :return: True if two input matrices are equivalent unitaries up to a global phase
    :rtype: bool
    """
    if not (is_unitary(unitary_op1) and is_unitary(unitary_op2)):
        return False

    nonzero = np.nonzero(unitary_op2)
    row = nonzero[0][0]
    column = nonzero[1][0]

    # differ by a global phase
    global_phase = unitary_op1[row, column] / unitary_op2[row, column]
    if np.allclose(unitary_op1, global_phase * unitary_op2) and np.allclose(
        np.abs(global_phase), 1.0
    ):
        return True
    else:
        return False
