# Copyright (c) 2022-2024 Quantum Bridge Technologies Inc.
# Copyright (c) 2022-2024 Ki3 Photonics Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
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
    :param pauli_type: which Pauli to look for: "x", "y" or "z", case-insensitively
    :type pauli_type: str
    :raises ValueError: if pauli_type is not one of "x", "y" or "z"
    :return:  a list containing the positions (row indices) of the Pauli operators of type pauli_type in and below
        the row specified by pivot[0] in the column given by pivot[1]
    :rtype: list
    """
    pauli = pauli_type.lower()
    if pauli == "x":
        match = (1, 0)
    elif pauli == "y":
        match = (1, 1)
    elif pauli == "z":
        match = (0, 1)
    else:
        # silently matching the identity rows here would, e.g., let
        # canonical_form return a tableau that is not in canonical form
        raise ValueError(
            f"pauli_type must be one of 'x', 'y' or 'z' (case-insensitive), got {pauli_type!r}"
        )
    n_row = np.shape(x_matrix)[0]

    pauli_list = []
    for row_i in range(pivot[0], n_row):
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
    n_row = np.shape(x_matrix)[0]
    # list of the rows (generators) with a pauli X operator in the pivot column
    pauli_x_list = []
    # list of the rows (generators) with a pauli Y operator in the pivot column
    pauli_y_list = []
    # list of the rows (generators) with a pauli Z operator in the pivot column
    pauli_z_list = []

    for row_i in range(pivot[0], n_row):
        if x_matrix[row_i, pivot[1]] == 1 and z_matrix[row_i, pivot[1]] == 0:
            pauli_x_list.append(row_i)
        elif x_matrix[row_i, pivot[1]] == 1 and z_matrix[row_i, pivot[1]] == 1:
            pauli_y_list.append(row_i)
        elif x_matrix[row_i, pivot[1]] == 0 and z_matrix[row_i, pivot[1]] == 1:
            pauli_z_list.append(row_i)

    return pauli_x_list, pauli_y_list, pauli_z_list


def _bare_z_row(tableau, qubit_position):
    """
    Find the generator that is a bare Pauli Z on the given qubit and the identity
    everywhere else. Such a generator is unique when it exists: two of them would be
    the same generator.

    :param tableau: the tableau to search
    :type tableau: StabilizerTableau
    :param qubit_position: the qubit the generator acts on
    :type qubit_position: int
    :return: the row index of that generator, or None if the tableau has none
    :rtype: int or None
    """
    for row_i in range(tableau.n_qubits):
        if (
            not tableau.x_matrix[row_i].any()
            and np.count_nonzero(tableau.z_matrix[row_i]) == 1
            and tableau.z_matrix[row_i, qubit_position] == 1
        ):
            return row_i
    return None


def inverse_circuit(tableau):
    """
    Find the inverse circuit that transforms the input stabilizer tableau back to the stabilizer tableau
    of :math:`|0\\rangle^{\\otimes n}` state

    :param tableau: the input tableau
    :type tableau: StabilizerTableau
    :return: the tableau that is converted to the basis state, list of gate instructions
    :rtype: StabilizerTableau, list[tuple]
    """
    n_qubits = tableau.n_qubits
    zero_state = StabilizerTableau(n_qubits)
    backup = tableau.copy()
    reduced, circuit_list = _reduce_to_zero_state(tableau, pivot_every_column=False)
    if reduced == zero_state:
        return reduced, circuit_list

    # The first Hadamard block can leave the later blocks without a pivot, or end on
    # |0...0> written with generators other than the bare Z on each qubit, which the
    # stabilizer inner product reads row by row. Start again from the input, in the
    # same object, with the block that places a pivot on every column.
    tableau.table = backup.table
    tableau.phase = backup.phase
    reduced, circuit_list = _reduce_to_zero_state(tableau, pivot_every_column=True)
    if reduced != zero_state:
        raise ValueError(
            "the tableau did not reduce to the all-zero state; it is not a valid "
            "stabilizer state"
        )
    return reduced, circuit_list


def _reduce_to_zero_state(tableau, pivot_every_column):
    """
    Reduce the tableau towards the tableau of :math:`|0\\rangle^{\\otimes n}`,
    with one of two versions of the first Hadamard block

    :param tableau: the input tableau, reduced in place
    :type tableau: StabilizerTableau
    :param pivot_every_column: False to apply H only to a Z-type column with support to
        its right; True to apply H to every column that has no X pivot and is not a
        bare-Z qubit, and then place a pivot on every diagonal entry
    :type pivot_every_column: bool
    :return: the reduced tableau, list of gate instructions
    :rtype: StabilizerTableau, list[tuple]
    """
    circuit_list = []
    n_qubits = tableau.n_qubits
    tableau = canonical_form(tableau)

    if pivot_every_column:
        # Hadamard block
        # The blocks below clear the X matrix by column operations, keying off the
        # diagonal entry of each column, and they leave a column alone when its qubit is
        # already reduced. So this block has to put a generator on every diagonal entry:
        # an X pivot on the qubits that still need work, the bare Z on the ones that do
        # not.
        #
        # The canonical form leaves the X matrix in reduced row echelon form, with the
        # generators that have no X support gathered in the rows underneath and their Z
        # parts reduced in turn. The leading entry of each row therefore names the
        # columns that carry an X pivot, and a generator that is a bare Z on a single
        # qubit says that qubit is already reduced -- unentangled, and in one of the two
        # computational basis states -- so no gate but the sign fix belongs on it. Every
        # other column without an X pivot gets a Hadamard, which moves that column's Z
        # entries into the X matrix. That makes the X matrix invertible on the columns
        # that need a pivot: a generator with no X support is orthogonal to the X row
        # space, because the group is abelian, and such an element supported only on the
        # pivot columns is the identity -- in reduced row echelon form the product
        # against pivot row i collapses to that element's entry at pivot column i, so
        # commutation kills the pivot coordinates one at a time. Their Z entries on the
        # other columns are therefore independent.
        x_pivot_columns = set()
        reduced_columns = set()
        for row_i in range(n_qubits):
            x_support = np.nonzero(tableau.x_matrix[row_i])[0]
            if x_support.size:
                x_pivot_columns.add(int(x_support[0]))
            elif np.count_nonzero(tableau.z_matrix[row_i]) == 1:
                reduced_columns.add(int(np.nonzero(tableau.z_matrix[row_i])[0][0]))

        for j in range(n_qubits):
            if j not in x_pivot_columns and j not in reduced_columns:
                circuit_list.append(("H", j))
                tableau = transform.hadamard_gate(tableau, j)

        # Move the generator for each column onto the diagonal and clear the column
        # below it. These are row operations: they change which generators are written
        # down, not the state, so they cost no gates. An already-reduced qubit takes its
        # own bare Z generator, which the eliminations cannot have touched -- they only
        # add into rows that have X support in the column being cleared.
        for j in range(n_qubits):
            if j in reduced_columns:
                pivot_row = _bare_z_row(tableau, j)
            else:
                rows_below = np.nonzero(tableau.x_matrix[j:n_qubits, j])[0]
                pivot_row = j + int(rows_below[0]) if rows_below.size else None
            if pivot_row is None or pivot_row < j:
                # Unreachable on a valid stabilizer state. It must raise rather than
                # assert: with assertions off, a pivot_row below j would swap a row
                # upward over an already-placed column and return a wrong tableau with a
                # circuit that does not zero the state -- silently, which is how this
                # defect survived in the first place.
                raise ValueError(
                    f"column {j} has no generator left to pivot on; the tableau is "
                    f"not a valid stabilizer state"
                )
            tableau = tab_row_swap(tableau, j, pivot_row)
            for row_i in range(j + 1, n_qubits):
                if tableau.x_matrix[row_i, j] == 1:
                    tableau = tab_row_sum(tableau, j, row_i)
    else:
        # Hadamard block
        pivot = [0, 0]
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
