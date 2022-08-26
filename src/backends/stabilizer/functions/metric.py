import numpy as np

from src.backends.stabilizer.functions.stabilizer import (
    canonical_form,
    inverse_circuit,
)
from src.backends.stabilizer.functions.linalg import row_sum
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
    tableau2 = sfc.run_circuit(tableau2, circ)

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
            identity_z = np.zeros(n_qubits)
            x_matrix = np.vstack((x1_matrix, identity_x)).astype(int)
            z_matrix = np.vstack((z1_matrix, identity_z)).astype(int)
            r_vector = np.zeros(n_qubits + 1)
            z_list = [j for j in range(n_qubits) if z2_matrix[i, j] == 1]
            for index in z_list:
                x_matrix, z_matrix, r_vector, _ = row_sum(
                    x_matrix,
                    z_matrix,
                    r_vector,
                    np.zeros(n_qubits + 1),
                    index,
                    n_qubits,
                )
            if (
                np.array_equal(x_matrix[-1], x2_matrix[i])
                and np.array_equal(z_matrix[-1], z2_matrix[i])
                and r_vector[-1] == 1
            ):
                return 0
    return 2 ** (-counter / 2)


def clifford_from_stabilizer(stabilizer_tableau):
    """

    :param stabilizer_tableau:
    :type stabilizer_tableau: StabilizerTableau
    :return:
    :rtype: CliffordTableau
    """
    n_qubits = stabilizer_tableau.n_qubits
    _, circuit = inverse_circuit(stabilizer_tableau)
    clifford_tableau = sfc.create_n_ket0_state(n_qubits)
    return sfc.run_circuit(clifford_tableau, circuit)
