import numpy as np
from src.backends.stabilizer.functions.linalg import hadamard_transform
from src.backends.stabilizer.functions.stabilizer import (
    canonical_form,
    row_sum,
    inverse_circuit,
)


def fidelity(tableau1, tableau2):
    """
    Compute the fidelity of two stabilizer states given their tableaux.

    :param tableau1:
    :type tableau1: CliffordTableau
    :param tableau2:
    :type tableau2: CliffordTableau
    :return:
    :rtype: float
    """
    return np.abs(inner_product(tableau1, tableau2)) ** 2


def inner_product(tableau1, tableau2):
    """
    Calculate the inner product of two stabilizer states using CliffordTableau

    :param tableau1:
    :type tableau1: CliffordTableau
    :param tableau2:
    :type tableau2: CliffordTableau
    :return:
    :rtype:
    """
    n_qubits = tableau1.n_qubits
    assert tableau1.n_qubits == tableau2.n_qubits
    x1_matrix = tableau1.stabilizer_x
    z1_matrix = tableau1.stabilizer_z
    x2_matrix = tableau2.stabilizer_x
    z2_matrix = tableau2.stabilizer_z
    r1_vector = tableau1.phase[n_qubits : 2 * n_qubits]
    r2_vector = tableau2.phase[n_qubits : 2 * n_qubits]
    _, _, _, circ1 = inverse_circuit(x1_matrix, z1_matrix, r1_vector)
    # apply inverse circuit on the 2nd state
    for ops in circ1:
        if ops[0] == "H":
            x2_matrix, z2_matrix = hadamard_transform(x2_matrix, z2_matrix, ops[1])
        if ops[0] == "P":
            z2_matrix[:, ops[1]] = z2_matrix[:, ops[1]] ^ x2_matrix[:, ops[1]]
        if ops[0] == "CNOT":
            x2_matrix[:, ops[2]] = x2_matrix[:, ops[2]] ^ x2_matrix[:, ops[1]]
            z2_matrix[:, ops[1]] = z2_matrix[:, ops[1]] ^ z2_matrix[:, ops[2]]
        if ops[0] == "CZ":
            x2_matrix, z2_matrix = hadamard_transform(x2_matrix, z2_matrix, ops[2])
            x2_matrix[:, ops[2]] = x2_matrix[:, ops[2]] ^ x2_matrix[:, ops[1]]
            z2_matrix[:, ops[1]] = z2_matrix[:, ops[1]] ^ z2_matrix[:, ops[2]]
            x2_matrix, z2_matrix = hadamard_transform(x2_matrix, z2_matrix, ops[2])
    # make the updated 2nd state canonical form
    x2_matrix, z2_matrix, r2_vector = canonical_form(x2_matrix, z2_matrix, r2_vector)
    counter = 0
    for i in range(n_qubits):
        if np.any(x2_matrix[i]):
            counter = counter + 1
        else:
            identity_x = np.zero(n_qubits)
            identity_y = np.zero(n_qubits)
            x_matrix = np.vstack((x1_matrix, identity_x))
            z_matrix = np.vstack((z1_matrix, identity_y))
            r_vector = np.zero(n_qubits + 1)
            z_list = [j for j in range(n_qubits) if z2_matrix[i, j] == 1]
            for index in z_list:
                x_matrix, z_matrix, r_vector = row_sum(
                    x_matrix, z_matrix, r_vector, index, n_qubits
                )
            if (
                np.array_equal(x_matrix[-1], x2_matrix[i])
                and np.array_equal(z_matrix[-1] == z2_matrix[i])
                and r_vector[-1] == 1
            ):
                return 0
    return 2 ** (-counter / 2)
