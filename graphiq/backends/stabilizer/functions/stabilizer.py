"""
Functions that are specific for StabilizerTableau
"""
import numpy as np

import graphiq.backends.stabilizer.functions.transformation as transform
from graphiq.backends.stabilizer.functions.linalg import (
    row_swap,
    row_sum,
)
from graphiq.backends.stabilizer.tableau import StabilizerTableau


def one_pauli_type_finder(x_matrix, z_matrix, pivot, pauli_type):
    """
    Find all row indices of the Pauli operators of a particular type that are present in and below the row given
    by pivot[0] and in the column specified by pivot[1] in the stabilizer tableau.

    :param x_matrix: X matrix in the symplectic representation of the stabilizers
    :type x_matrix: np.ndarray
    :param z_matrix: Z matrix in the symplectic representation of the stabilizers
    :type z_matrix: np.ndarray
    :param pivot: the location of the pivot element [i,j] on the i-th row and j-th column of the stabilizer tableau
    :type pivot: list
    :param pauli_type: A string to specify the type of Pauli to be found
    :type pauli_type: str
    :return:  a list containing the positions (row indices) of the Pauli operators of type pauli_type in and below
        the row specified by pivot[0] in the column given by pivot[1]
    :rtype: list
    """
    if pauli_type.lower() == "x":
        match = (1, 0)
    elif pauli_type.lower() == "y":
        match = (1, 1)
    elif pauli_type.lower() == "z":
        match = (0, 1)
    else:
        match = (0, 0)
    n_qubits = np.shape(x_matrix)[1]

    pauli_list = []
    for row_i in range(pivot[0], n_qubits):
        if (x_matrix[row_i, pivot[1]], z_matrix[row_i, pivot[1]]) == match:
            pauli_list.append(row_i)

    return pauli_list


def pauli_type_finder(x_matrix, z_matrix, pivot):
    """
    Find all row indices of the Pauli operators of each type that are present in and below the row given
    by pivot[0] and in the column specified by pivot[1] in the stabilizer tableau.

    :param x_matrix: x matrix in the symplectic representation of the stabilizers
    :type x_matrix: np.ndarray
    :param z_matrix: z matrix in the symplectic representation of the stabilizers
    :type z_matrix: np.ndarray
    :param pivot: the location of the pivot element [i,j] on the i-th row and j-th column of the stabilizer tableau
    :type pivot: list
    :return: three lists each containing the positions (row indices) of the Pauli X, Y, and Z operators in and below
        the row specified by pivot[0] in the column given by pivot[1], e.g. if the first list is [3, 4],
        it means there are Pauli X operators in rows 3 and 4 in the pivot column.
    :rtype: list, list, list
    """
    n_qubits = np.shape(x_matrix)[1]
    # list of the rows (generators) with a pauli X operator in the pivot column
    pauli_x_list = []
    # list of the rows (generators) with a pauli Y operator in the pivot column
    pauli_y_list = []
    # list of the rows (generators) with a pauli Z operator in the pivot column
    pauli_z_list = []

    for row_i in range(pivot[0], n_qubits):
        if x_matrix[row_i, pivot[1]] == 1 and z_matrix[row_i, pivot[1]] == 0:
            pauli_x_list.append(row_i)
        elif x_matrix[row_i, pivot[1]] == 1 and z_matrix[row_i, pivot[1]] == 1:
            pauli_y_list.append(row_i)
        elif x_matrix[row_i, pivot[1]] == 0 and z_matrix[row_i, pivot[1]] == 1:
            pauli_z_list.append(row_i)

    return pauli_x_list, pauli_y_list, pauli_z_list


def inverse_circuit(tableau):
    """
    Find the inverse circuit that transforms the input stabilizer tableau back to the stabilizer tableau
    of :math:`|0\\rangle^{\\otimes n}` state

    :param tableau: the input tableau
    :type tableau: StabilizerTableau
    :return: the tableau that is converted to the basis state, list of gate instructions
    :rtype: StabilizerTableau, list[tuple]
    """
    circuit_list = []
    pivot = [0, 0]
    n_qubits = tableau.n_qubits
    tableau = canonical_form(tableau)

    # Hadamard block
    for j in range(n_qubits):
        pivot[1] = j
        x_list, y_list, z_list = pauli_type_finder(
            tableau.x_matrix, tableau.z_matrix, pivot
        )
        if x_list:
            tableau = tab_row_swap(tableau, pivot[0], x_list[0])
        elif y_list:
            tableau = tab_row_swap(tableau, pivot[0], y_list[0])
        elif z_list:
            tableau = tab_row_swap(tableau, pivot[0], z_list[-1])
            if np.any(tableau.x_matrix[pivot[0], j + 1 : n_qubits]) or np.any(
                tableau.z_matrix[pivot[0], j + 1 : n_qubits]
            ):
                circuit_list.append(("H", j))
                tableau = transform.hadamard_gate(tableau, j)
        pivot[0] = pivot[0] + 1
    # CNOT block
    for j in range(n_qubits):
        for k in range(j + 1, n_qubits):
            if tableau.x_matrix[j, k] == 1:
                circuit_list.append(("CNOT", j, k))
                tableau = transform.cnot_gate(tableau, j, k)

    # CZ block
    for j in range(n_qubits):
        for k in range(j + 1, n_qubits):
            if tableau.x_matrix[j, k] == 0 and tableau.z_matrix[j, k] == 1:
                circuit_list.append(("CZ", j, k))
                tableau = transform.control_z_gate(tableau, j, k)

    # Phase gate block
    for j in range(n_qubits):
        if tableau.x_matrix[j, j] == 1 and tableau.z_matrix[j, j] == 1:
            # add the phase gate to the circuit list
            circuit_list.append(("P", j))
            tableau = transform.phase_gate(tableau, j)

    # Hadamard block
    for j in range(n_qubits):
        if tableau.x_matrix[j, j] == 1 and tableau.z_matrix[j, j] == 0:
            circuit_list.append(("H", j))
            tableau = transform.hadamard_gate(tableau, j)
    # Eliminate Zs
    for j in range(n_qubits):
        for k in range(j + 1, n_qubits):
            if tableau.x_matrix[k, j] == 0 and tableau.z_matrix[k, j] == 1:
                tableau = tab_row_sum(tableau, j, k)

    # Eliminate phase
    for i in np.nonzero(tableau.phase)[0]:
        tableau = transform.x_gate(tableau, i)
        circuit_list.append(("X", int(i)))
    return tableau, circuit_list


def tab_row_sum(tableau, row_to_add, target_row):
    """
    Takes the full stabilizer tableau as input and sets the stabilizer generator in the target_row equal to
    (row_to_add + target_row) while updating the phase vector.
    This is based on the section III of the article arXiv:quant-ph/0406196v5

    :param tableau: the stabilizer tableau
    :type tableau: StabilizerTableau
    :param row_to_add: the stabilizer generator to multiply the target stabilizer generator with
    :type row_to_add: int
    :param target_row: the stabilizer generator to be multiplied by the "to_add" stabilizer generator
    :type target_row: int
    :return: updated stabilizer tableau
    :rtype: StabilizerTableau
    """
    n_qubits = tableau.n_qubits
    x_matrix, z_matrix, r_vector, _ = row_sum(
        tableau.x_matrix,
        tableau.z_matrix,
        tableau.phase,
        np.zeros(n_qubits),
        row_to_add,
        target_row,
    )

    tableau.x_matrix = x_matrix
    tableau.z_matrix = z_matrix
    tableau.phase = r_vector
    return tableau


def tab_row_swap(tableau, first_row, second_row):
    """
    swaps the rows of the full stabilizer tableau (including the phase factor vector)

    :param tableau: the input tableau to be manipulated
    :type tableau: StabilizerTableau
    :param first_row: one of the rows to be swapped
    :type first_row: int
    :param second_row: the other row to be swapped
    :type second_row: int
    :return: updated stabilizer tableau
    :rtype: StabilizerTableau
    """
    tableau.x_matrix = row_swap(tableau.x_matrix, first_row, second_row)
    tableau.z_matrix = row_swap(tableau.z_matrix, first_row, second_row)
    tableau.phase = row_swap(tableau.phase, first_row, second_row)
    return tableau


def _process_one_pauli(tableau, pivot, pauli_list):
    """
    Helper function to process one Pauli list

    :param tableau: the input tableau to be processed
    :type tableau: StabilizerTableau
    :param pivot: a pivot position (row index, column index)
    :type pivot: [int, int]
    :param pauli_list: a list of positions to apply actions
    :type pauli_list: list
    :return: the tableau after processing and the updated pivot position
    :rtype: StabilizerTableau, [int, int]
    """
    tableau = tab_row_swap(tableau, pivot[0], pauli_list[0])

    # remove the first element of the list
    pauli_list = pauli_list[1:]

    for row_i in pauli_list:
        # multiplying rows with similar pauli to eliminate them
        tableau = tab_row_sum(tableau, pivot[0], row_i)

    pivot = [pivot[0] + 1, pivot[1] + 1]
    return tableau, pivot


def _process_two_pauli(
    tableau,
    pivot,
    pauli_list_dict,
    pauli_type1,
    pauli_type2,
):
    """
    Helper function to process two Pauli lists

    :param tableau: the input tableau to be processed
    :type tableau: StabilizerTableau
    :param pivot: a pivot position (row index, column index)
    :type pivot: [int, int]
    :param pauli_list_dict: a dictionary that contains all Pauli lists (for X, Y, Z)
    :type pauli_list_dict: dict
    :param pauli_type1: a string that identifies which Pauli to process
    :type pauli_type1: str
    :param pauli_type2: a string that identifies which Pauli to process
    :type pauli_type2: str
    :return: the tableau after processing, the updated pivot position
    :rtype: StabilizerTableau, [int, int]
    """

    # swap the pivot and its next row with them

    tableau = tab_row_swap(tableau, pivot[0], pauli_list_dict[pauli_type1][0])
    # update pauli lists
    pauli_x_list, pauli_y_list, pauli_z_list = pauli_type_finder(
        tableau.x_matrix, tableau.z_matrix, pivot
    )

    pauli_list_dict["x"] = pauli_x_list
    pauli_list_dict["y"] = pauli_y_list
    pauli_list_dict["z"] = pauli_z_list
    tableau = tab_row_swap(tableau, pivot[0] + 1, pauli_list_dict[pauli_type2][0])
    # update pauli lists
    pauli_x_list, pauli_y_list, pauli_z_list = pauli_type_finder(
        tableau.x_matrix, tableau.z_matrix, pivot
    )
    pauli_list_dict["x"] = pauli_x_list
    pauli_list_dict["y"] = pauli_y_list
    pauli_list_dict["z"] = pauli_z_list
    assert (
        pauli_list_dict[pauli_type1][0] == pivot[0]
        and pauli_list_dict[pauli_type2][0] == pivot[0] + 1
    ), "row operations failed"

    # remove the first element of the list
    pauli_list_dict[pauli_type1] = pauli_list_dict[pauli_type1][1:]
    pauli_list_dict[pauli_type2] = pauli_list_dict[pauli_type2][1:]

    for row_i in pauli_list_dict[pauli_type1]:
        # multiplying rows with similar pauli to eliminate them
        tableau = tab_row_sum(tableau, pivot[0], row_i)

    for row_j in pauli_list_dict[pauli_type2]:
        # multiplying rows with similar pauli to eliminate them
        tableau = tab_row_sum(tableau, pivot[0] + 1, row_j)

    pivot = [pivot[0] + 2, pivot[1] + 1]
    return tableau, pivot


def one_step_rref(tableau, pivot):
    """
    ROW-REDUCED ECHELON FORM algorithm that takes the pivot element location and stabilizer tableau,
    and converts the elements below the pivot to the standard row echelon form.
    This is one of the steps of the full row reduced echelon form algorithm.

    :param tableau: the input tableau to be processed
    :type tableau: StabilizerTableau
    :param pivot: the location of the pivot element [i,j] on the i-th row and j-th column of the stabilizer tableau.
    :type pivot: [int, int]
    :return: updated stabilizer tableau and updated pivot
    :rtype: StabilizerTableau, list
    """

    # pauli_x_list  = list of the rows (generators) with a pauli X operator in the pivot column
    # pauli_y_list  = list of the rows (generators) with a pauli Y operator in the pivot column
    # pauli_z_list  = list of the rows (generators) with a pauli Z operator in the pivot column
    pauli_x_list, pauli_y_list, pauli_z_list = pauli_type_finder(
        tableau.x_matrix, tableau.z_matrix, pivot
    )
    pauli_list_dict = {"x": pauli_x_list, "y": pauli_y_list, "z": pauli_z_list}
    # case of no pauli operator
    if not (pauli_x_list or pauli_y_list or pauli_z_list):
        pivot = [pivot[0], pivot[1] + 1]
        return tableau, pivot

    # case of only 1 kind of pauli
    elif pauli_x_list and (not pauli_y_list) and (not pauli_z_list):  # only X
        return _process_one_pauli(tableau, pivot, pauli_x_list)

    elif pauli_y_list and (not pauli_x_list) and (not pauli_z_list):  # only Y
        return _process_one_pauli(tableau, pivot, pauli_y_list)

    elif pauli_z_list and (not pauli_x_list) and (not pauli_y_list):  # only Z
        return _process_one_pauli(tableau, pivot, pauli_z_list)

    # case of two kinds of pauli
    elif not pauli_x_list:  # pauli y and z exist in the column below pivot
        return _process_two_pauli(tableau, pivot, pauli_list_dict, "y", "z")

    elif not pauli_y_list:  # pauli x and z exist in the column below pivot
        return _process_two_pauli(tableau, pivot, pauli_list_dict, "x", "z")

    elif not pauli_z_list:  # pauli x and y exist in the column below pivot
        return _process_two_pauli(tableau, pivot, pauli_list_dict, "x", "y")

    # case of all three kinds of paulis available in the column
    else:
        tableau, _ = _process_two_pauli(tableau, pivot, pauli_list_dict, "x", "z")
        # update pauli lists
        pauli_x_list, pauli_y_list, pauli_z_list = pauli_type_finder(
            tableau.x_matrix, tableau.z_matrix, pivot
        )
        for row_k in pauli_y_list:
            # multiplying the pauli Y with pauli X to make it Z
            tableau = tab_row_sum(tableau, pivot[0], row_k)
            # multiplying the now Z row with another Z to eliminate it
            tableau = tab_row_sum(tableau, pivot[0] + 1, row_k)
        pivot = [pivot[0] + 2, pivot[1] + 1]
        return tableau, pivot


def rref(tableau):
    """
    Takes stabilizer tableau, and converts it to the standard row echelon form.
    This implements the algorithm in New J. Phys. 7, 170 (2005).

    :param tableau: the input tableau to be processed
    :type tableau: StabilizerTableau
    :return: stabilizer tableau in the row reduced echelon form
    :rtype: StabilizerTableau
    """

    pivot = [0, 0]
    n_qubits = tableau.n_qubits
    while pivot[0] <= n_qubits - 1 and pivot[1] <= n_qubits - 1:
        tableau, pivot = one_step_rref(tableau, pivot)
    # rank check
    assert (
        pivot[0] >= n_qubits - 1
    ), "Invalid input. One of the stabilizers is identity on all qubits!"
    return tableau


def canonical_form(tableau):
    """
    Takes stabilizer tableau, and converts it to the canonical reduced echelon form, which is
    different from standard row reduced echelon form.

    :param tableau: the input tableau to be processed
    :type tableau: StabilizerTableau
    :return: stabilizer tableau in the canonical form
    :rtype: StabilizerTableau
    """
    pivot = [0, 0]
    n_qubits = tableau.n_qubits
    # setup X block
    for j in range(n_qubits):
        pivot[1] = j

        x_list, y_list, z_list = pauli_type_finder(
            tableau.x_matrix, tableau.z_matrix, pivot
        )
        if x_list or y_list:
            if x_list:
                tableau = tab_row_swap(tableau, pivot[0], x_list[0])

            else:
                tableau = tab_row_swap(tableau, pivot[0], y_list[0])

            for row_m in range(n_qubits):
                if tableau.x_matrix[row_m, j] == 1 and row_m != pivot[0]:
                    # update the generator in row_m
                    tableau = tab_row_sum(tableau, pivot[0], row_m)
            pivot[0] = pivot[0] + 1

    # setup Z block
    for j in range(n_qubits):
        pivot[1] = j
        z_list = one_pauli_type_finder(tableau.x_matrix, tableau.z_matrix, pivot, "z")
        if z_list:
            tableau = tab_row_swap(tableau, pivot[0], z_list[0])

            for row_m in range(n_qubits):
                if tableau.z_matrix[row_m, j] == 1 and row_m != pivot[0]:
                    # update the generator in row_m
                    tableau = tab_row_sum(tableau, pivot[0], row_m)
            pivot[0] = pivot[0] + 1
    # confirm if there is any trivial rows equivalent to all identity matrices.

    assert pivot[0] == n_qubits
    return tableau


def insert_qubit(tableau, new_position):
    """
    Insert a qubit in :math:`| 0 \\rangle` state to a given position.

    :param tableau: the state represented by a StabilizerTableau before insertion
    :type tableau: StabilizerTableau
    :param new_position: the future position of the inserted qubit
    :type new_position: int
    :return: updated state
    :rtype: StabilizerTableau
    """
    n_qubits = tableau.n_qubits
    assert new_position <= n_qubits
    new_column = np.zeros(n_qubits)
    new_row = np.zeros(n_qubits + 1)

    # x  part
    tmp_x = np.insert(tableau.x_matrix, new_position, new_column, axis=1)
    tmp_x = np.insert(tmp_x, new_position, new_row, axis=0)

    # z  part
    tmp_z = np.insert(tableau.z_matrix, new_position, new_column, axis=1)
    tmp_z = np.insert(tmp_z, new_position, new_row, axis=0)

    # phase vector part
    new_phase = np.insert(tableau.phase, new_position, 0)

    tableau.expand(np.hstack([tmp_x, tmp_z]), new_phase)

    # set the new qubit to ket 0 state
    tableau.z_matrix[new_position, new_position] = 1

    return tableau
