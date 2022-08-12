import numpy as np
from src.backends.stabilizer.functions.linalg import hadamard_transform
from src.backends.stabilizer.functions.stabilizer import (
    canonical_form,
    row_sum,
    inverse_circuit,
)
import src.backends.stabilizer.functions.clifford as sfc


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
    stabilizer_tableau1 = tableau1.to_stabilizer()
    _, circ = inverse_circuit(stabilizer_tableau1)

    # apply inverse circuit on the 2nd state
    for ops in circ:
        if ops[0] == "H":
            tableau2 = sfc.hadamard_gate(tableau2, ops[1])
        if ops[0] == "P":
            tableau2 = sfc.phase_gate(tableau2, ops[1])
        if ops[0] == "CNOT":
            tableau2 = sfc.cnot_gate(tableau2, ops[1], ops[2])
        if ops[0] == "CZ":
            tableau2 = sfc.control_z_gate(tableau2, ops[1], ops[2])

    # make the updated 2nd state canonical form
    stabilizer_tableau2 = tableau2.to_stabilizer()
    stabilizer_tableau2 = canonical_form(stabilizer_tableau2)
    x1_matrix = tableau1.stabilizer_x
    z1_matrix = tableau1.stabilizer_z
    x2_matrix = stabilizer_tableau2.x_matrix
    z2_matrix = stabilizer_tableau2.z_matrix
    counter = 0
    for i in range(n_qubits):
        if np.any(x2_matrix[i]):
            counter = counter + 1
        else:
            identity_x = np.zeros(n_qubits)
            identity_y = np.zeros(n_qubits)
            x_matrix = np.vstack((x1_matrix, identity_x))
            z_matrix = np.vstack((z1_matrix, identity_y))
            r_vector = np.zeros(n_qubits + 1)
            z_list = [j for j in range(n_qubits) if z2_matrix[i, j] == 1]
            for index in z_list:
                x_matrix, z_matrix, r_vector = row_sum(
                    x_matrix, z_matrix, r_vector, index, n_qubits
                )
            if (
                np.array_equal(x_matrix[-1], x2_matrix[i])
                and np.array_equal(z_matrix[-1], z2_matrix[i])
                and r_vector[-1] == 1
            ):
                return 0
    return 2 ** (-counter / 2)
