import numpy as np


def row_swap(input_matrix, first_row, second_row):
    """
    Swap two rows of a matrix

    :param input_matrix: a matrix
    :type input_matrix: numpy.ndarray
    :param first_row: the first row
    :type first_row: int
    :param second_row: the second row
    :type second_row: int
    :return: the matrix after swapping those two row
    :rtype: numpy.ndarray
    """
    input_matrix[[first_row, second_row]] = input_matrix[[second_row, first_row]]
    return input_matrix


def add_rows(input_matrix, row_to_add, resulting_row):
    """
    Add two rows together modulo 2 and put it in the row of the second input

    :param input_matrix: a binary matrix
    :type input_matrix: numpy.ndarray
    :param row_to_add: the index of the row to add
    :type row_to_add: int
    :param resulting_row: the index of the row where the result is put
    :type resulting_row: int
    :return: the matrix after adding two rows modulo 2 and putting in the row of the second input
    :rtype: numpy.ndarray
    """
    input_matrix[resulting_row] = (
        input_matrix[row_to_add] + input_matrix[resulting_row]
    ) % 2
    return input_matrix


def column_swap(input_matrix, first_col, second_col):
    """
    Swap two columns of a matrix

    :param input_matrix: a matrix
    :type input_matrix: numpy.ndarray
    :param first_col: the first column
    :type first_col: int
    :param second_col: the second column
    :type second_col: int
    :return: the matrix after swapping those two columns
    :rtype: numpy.ndarray
    """
    input_matrix[:, [first_col, second_col]] = input_matrix[:, [second_col, first_col]]
    return input_matrix


def add_columns(input_matrix, col_to_add, resulting_col):
    """
    Add two rows together modulo 2 and put it in the row of the second input

    :param input_matrix: a binary matrix
    :type input_matrix: numpy.ndarray
    :param col_to_add: the index of the column to add
    :type col_to_add: int
    :param resulting_col: the index of the column where the result is put
    :type resulting_col: int
    :return: the matrix after adding two column modulo 2 and putting in the column of the second input
    :rtype: numpy.ndarray
    """
    input_matrix[:, resulting_col] = (
        input_matrix[:, col_to_add] + input_matrix[:, resulting_col]
    ) % 2
    return input_matrix


def multiply_columns(matrix_one, matrix_two, first_col, second_col):
    """
    Multiplies two columns of possibly two matrices (element-wise), and returns an array containing the result.

    :param matrix_one: a matrix
    :type matrix_one: numpy.ndarray
    :param matrix_two: a second matrix of the same number of rows as the first one
    :type matrix_two: numpy.ndarray
    :param first_col: index of the column to be used from the first matrix
    :type first_col: int
    :param second_col: index of the column to be used from the second matrix
    :type second_col: int
    :raises AssertionError: if two matrices have different number of rows
        or the specified column index is out of range in one of the matrices
    :return: the resulting 1-d array of length n (= number of the rows of the matrices)
    :rtype: numpy.ndarray
    """
    n_rows, _ = np.shape(matrix_one)
    assert np.shape(matrix_one)[0] == np.shape(matrix_two)[0]

    assert (
        first_col < np.shape(matrix_one)[1] and second_col < np.shape(matrix_two)[1]
    ), "the specified column index is out of range in one of the matrices"

    resulting_col = np.multiply(matrix_one[:, first_col], matrix_two[:, second_col])
    # reshape into column form:
    # resulting_col = resulting_col.reshape(n_rows, 1)
    return resulting_col


def hadamard_transform(x_matrix, z_matrix, positions):
    """
    Apply a Hadamard gate on each qubit specified by positions. This action is equivalent to a
    column swap between X matrix and Z matrix for the corresponding qubits.
    (not a stabilizer backend quantum gate, just a helper function)

    :param x_matrix: X part of the symplectic representation
    :type x_matrix: numpy.ndarray
    :param z_matrix: Z part of the symplectic representation
    :type z_matrix: numpy.ndarray
    :param positions: positions of qubits where the Hadamard gates are applied
    :type positions: list[int]
    :rtype: numpy.ndarray, numpy.ndarray
    :return: the resulting X matrix and Z matrix
    """
    temp1 = list(z_matrix[:, positions])
    temp2 = list(x_matrix[:, positions])
    z_matrix[:, positions] = temp2
    x_matrix[:, positions] = temp1
    return x_matrix, z_matrix


def row_reduction(x_matrix, z_matrix):
    """
    Turns the x_matrix into a row reduced echelon form. Applies same row operations on z_matrix.

    :param x_matrix: binary matrix for representing Pauli X part of the symplectic binary
        representation of the stabilizer generators.
    :type x_matrix: numpy.ndarray
    :param z_matrix:binary matrix for representing Pauli Z part of the
        symplectic binary representation of the stabilizer generators
    :type z_matrix: numpy.ndarray
    :return: a tuple of the transformed x_matrix and z_matrix and the index of the last non-zero row of the new x_matrix
    :rtype: tuple(numpy.ndarray, numpy.ndarray, int)
    """

    pivot = [0, 0]
    old_pivot = [1, 1]

    while pivot[1] != old_pivot[1]:
        # all row reduction operations will at least change the column of the pivot by 1 (not true for its row! due
        # to last column pivot)
        old_pivot = pivot
        x_matrix, z_matrix, pivot = _row_red_one_step(x_matrix, z_matrix, pivot)
    return x_matrix, z_matrix, pivot[0]


def _row_red_one_step(x_matrix, z_matrix, pivot):
    """
    A helper function to apply one step of the row reduction algorithm, only on the pivot provided here.
    It is used in the main row reduction function.

    :param x_matrix: binary matrix for representing Pauli X part of the symplectic binary
        representation of the stabilizer generators
    :type x_matrix: numpy.ndarray
    :param z_matrix:binary matrix for representing Pauli Z part of the
        symplectic binary representation of the stabilizer generators
    :type z_matrix: numpy.ndarray
    :param pivot: a location in the input matrix
    :type pivot: list[int]
    :return: a tuple of the transformed x_matrix and z_matrix and the new pivot location
    :rtype: tuple(numpy.ndarray, numpy.ndarray, list[int])
    """
    n_row, n_column = np.shape(x_matrix)
    if pivot[1] == (n_column - 1):
        the_ones = []
        for i in range(pivot[0], n_row):
            if x_matrix[i, pivot[1]] == 1:
                the_ones.append(i)
        if not the_ones:
            # empty under (and including) pivot element on last column
            pivot[0] = pivot[0] - 1
        else:
            x_matrix = row_swap(x_matrix, the_ones[0], pivot[0])
            z_matrix = row_swap(z_matrix, the_ones[0], pivot[0])
            the_ones.remove(the_ones[0])
            for j in the_ones:
                x_matrix = add_rows(x_matrix, pivot[0], j)
                z_matrix = add_rows(z_matrix, pivot[0], j)
        return x_matrix, z_matrix, pivot
    elif pivot[0] == (n_row - 1):
        if x_matrix[pivot[0], pivot[1]] == 1:
            return x_matrix, z_matrix, pivot
        else:
            pivot = [pivot[0], pivot[1] + 1]
            return x_matrix, z_matrix, pivot

    else:
        # list of rows with value 1 under the pivot element
        the_ones = []
        for i in range(pivot[0], n_row):
            if x_matrix[i, pivot[1]] == 1:
                the_ones.append(i)
        # check if the column below is empty to skip it
        if not the_ones:
            pivot = [pivot[0], pivot[1] + 1]
            return x_matrix, z_matrix, pivot
        else:
            x_matrix = row_swap(x_matrix, the_ones[0], pivot[0])
            z_matrix = row_swap(z_matrix, the_ones[0], pivot[0])
            the_ones.remove(the_ones[0])
            for j in the_ones:
                x_matrix = add_rows(x_matrix, pivot[0], j)
                z_matrix = add_rows(z_matrix, pivot[0], j)
            pivot = [pivot[0] + 1, pivot[1] + 1]
            return x_matrix, z_matrix, pivot


def g_function(x_1, z_1, x_2, z_2):
    """
    A helper function to use in rowsum function. Takes 4 bits (2 pauli matrices in binary representation) as input and
    returns the phase factor needed when the two Pauli matrices are multiplied: Pauli_1 * Pauli_2

    Refer to section III of arXiv:quant-ph/0406196v5

    :param x_1: the x bit of the first Pauli operator
    :type x_1: int
    :param z_1: the z bit of the first Pauli operator
    :type z_1: int
    :param x_2: the x bit of the second Pauli operator
    :type x_2: int
    :param z_2: the z bit of the second Pauli operator
    :type z_2: int
    :return: the exponent k in the phase factor: i^k where "i" is the unit imaginary number.
    :rtype: int
    """
    if not (x_1 or z_1):  # both equal to zero
        return 0
    if x_1 and z_1:
        return (z_2 - x_2) % 4
    if x_1 == 1 and z_1 == 0:
        return (z_2 * (2 * x_2 - 1)) % 4
    if x_1 == 0 and z_1 == 1:
        return (x_2 * (1 - 2 * z_2)) % 4


def row_sum(x_matrix, z_matrix, r_vector, row_to_add, target_row):
    """
    Takes the full stabilizer tableau as input and sets the stabilizer generator in the target_row equal to
    (row_to_add + target_row) while keeping track  of the phase factor by updating the r_vector.
    This is based on the section III of the article arXiv:quant-ph/0406196v5

    :param x_matrix: x matrix in the symplectic representation of the stabilizers
    :type x_matrix: np.ndarray
    :param z_matrix: z matrix in the symplectic representation of the stabilizers
    :type z_matrix: np.ndarray
    :param r_vector: the vector of phase factors.
    :type r_vector: np.ndarray
    :param row_to_add: the stabilizer to multiply the target stabilizer with
    :type row_to_add: int
    :param target_row: the stabilizer to be multiplied by the "to_add" stabilizer
    :type target_row: int
    :return: updated stabilizer tableau
    :rtype: numpy.ndarray, numpy.ndarray, numpy.ndarray
    """
    n_qubits = np.shape(x_matrix)[1]
    # determining the phase factor
    g_sum = 0
    for j in range(n_qubits):
        g_sum = g_sum + g_function(
            x_matrix[row_to_add, j],
            z_matrix[row_to_add, j],
            x_matrix[target_row, j],
            z_matrix[target_row, j],
        )
    if (2 * r_vector[target_row] + 2 * r_vector[row_to_add] + g_sum) % 4 == 0:
        r_vector[target_row] = 0
    elif (2 * r_vector[target_row] + 2 * r_vector[row_to_add] + g_sum) % 4 == 2:
        r_vector[target_row] = 1
    else:
        raise Exception("input cannot be valid, due to unexpected outcome")

    # calculating the resulting new matrices after adding row i to h.
    x_matrix = add_rows(x_matrix, row_to_add, target_row)
    z_matrix = add_rows(z_matrix, row_to_add, target_row)

    return x_matrix, z_matrix, r_vector


def row_swap_full(x_matrix, z_matrix, r_vector, first_row, second_row):
    """
    swaps the rows of the full stabilizer tableau (including the phase factor vector)

    :param x_matrix: x matrix in the symplectic representation of the stabilizers
    :type x_matrix: np.ndarray
    :param z_matrix: z matrix in the symplectic representation of the stabilizers
    :type z_matrix: np.ndarray
    :param r_vector: the vector of phase factors.
    :type r_vector: np.ndarray
    :param first_row: one of the rows to be swapped
    :type first_row: int
    :param second_row: the other row to be swapped
    :type second_row: int
    :return: updated stabilizer tableau
    :rtype: numpy.ndarray, numpy.ndarray, numpy.ndarray
    """
    x_matrix = row_swap(x_matrix, first_row, second_row)
    z_matrix = row_swap(z_matrix, first_row, second_row)
    r_vector = row_swap(r_vector, first_row, second_row)
    return x_matrix, z_matrix, r_vector


def pauli_type_finder(x_matrix, z_matrix, pivot):
    """
    A function that counts the types and the number of the Pauli operators that are present on and below an element
    (the pivot) in the stabilizer tableau.

    :param x_matrix: x matrix in the symplectic representation of the stabilizers
    :type x_matrix: np.ndarray
    :param z_matrix: z matrix in the symplectic representation of the stabilizers
    :type z_matrix: np.ndarray
    :param pivot: the location of the pivot element [i,j] on the i-th row and j-th column of the stabilizer tableau.
    :type pivot: list
    :return: three lists each containing the positions (row indices) of the Pauli X, Y, and Z operators below the pivot,
    for example, if the first list is [3, 4] it means there are Pauli X operators in rows 3 and 4 in the pivot column.
    :rtype: list, list, list
    """
    n_qubits = np.shape(x_matrix)[0]
    # list of the rows (generators) with a pauli X operator in the pivot column
    pauli_x_list = []
    # list of the rows (generators) with a pauli Y operator in the pivot column
    pauli_y_list = []
    # list of the rows (generators) with a pauli Z operator in the pivot column
    pauli_z_list = []

    for row_i in range(pivot[0], n_qubits):
        if x_matrix[row_i, pivot[1]] == 1 and z_matrix[row_i, pivot[1]] == 0:
            pauli_x_list.append(row_i)
        if x_matrix[row_i, pivot[1]] == 1 and z_matrix[row_i, pivot[1]] == 1:
            pauli_y_list.append(row_i)
        if x_matrix[row_i, pivot[1]] == 0 and z_matrix[row_i, pivot[1]] == 1:
            pauli_z_list.append(row_i)

    return pauli_x_list, pauli_y_list, pauli_z_list


def _process_one_pauli(x_matrix, z_matrix, r_vector, pivot, pauli_list):
    """
    Helper function to process one Pauli list

    :param x_matrix:
    :param z_matrix:
    :param r_vector:
    :param pivot:
    :param pauli_list:
    :return:
    """
    x_matrix, z_matrix, r_vector = row_swap_full(
        x_matrix, z_matrix, r_vector, pivot[0], pauli_list[0]
    )

    # remove the first element of the list
    pauli_list = pauli_list[1:]

    for row_i in pauli_list:
        # multiplying rows with similar pauli to eliminate them
        x_matrix, z_matrix, r_vector = row_sum(
            x_matrix, z_matrix, r_vector, pivot[0], row_i
        )

    pivot = [pivot[0] + 1, pivot[1] + 1]
    return x_matrix, z_matrix, r_vector, pivot


def _process_two_pauli(
    x_matrix,
    z_matrix,
    r_vector,
    pivot,
    pauli_list_dict,
    first_list_symbol,
    second_list_symbol,
):
    """
    Helper function to process two Pauli lists

    :param x_matrix:
    :param z_matrix:
    :param r_vector:
    :param pivot:
    :param pauli_list_dict:
    :param first_list_symbol:
    :param second_list_symbol:
    :return:
    """
    # swap the pivot and its next row with them

    x_matrix, z_matrix, r_vector = row_swap_full(
        x_matrix, z_matrix, r_vector, pivot[0], pauli_list_dict[first_list_symbol][0]
    )
    # update pauli lists
    pauli_x_list, pauli_y_list, pauli_z_list = pauli_type_finder(
        x_matrix, z_matrix, pivot
    )

    pauli_list_dict["x"] = pauli_x_list
    pauli_list_dict["y"] = pauli_y_list
    pauli_list_dict["z"] = pauli_z_list
    x_matrix, z_matrix, r_vector = row_swap_full(
        x_matrix,
        z_matrix,
        r_vector,
        pivot[0] + 1,
        pauli_list_dict[second_list_symbol][0],
    )
    # update pauli lists
    pauli_x_list, pauli_y_list, pauli_z_list = pauli_type_finder(
        x_matrix, z_matrix, pivot
    )
    pauli_list_dict["x"] = pauli_x_list
    pauli_list_dict["y"] = pauli_y_list
    pauli_list_dict["z"] = pauli_z_list
    assert (
        pauli_list_dict[first_list_symbol][0] == pivot[0]
        and pauli_list_dict[second_list_symbol][0] == pivot[0] + 1
    ), "row operations failed"

    # remove the first element of the list
    pauli_list_dict[first_list_symbol] = pauli_list_dict[first_list_symbol][1:]
    pauli_list_dict[second_list_symbol] = pauli_list_dict[second_list_symbol][1:]

    for row_i in pauli_list_dict[first_list_symbol]:
        # multiplying rows with similar pauli to eliminate them
        x_matrix, z_matrix, r_vector = row_sum(
            x_matrix, z_matrix, r_vector, pivot[0], row_i
        )

    for row_j in pauli_list_dict[second_list_symbol]:
        # multiplying rows with similar pauli to eliminate them
        x_matrix, z_matrix, r_vector = row_sum(
            x_matrix, z_matrix, r_vector, pivot[0] + 1, row_j
        )

    pivot = [pivot[0] + 2, pivot[1] + 1]
    return x_matrix, z_matrix, r_vector, pivot


def one_step_rref(x_matrix, z_matrix, r_vector, pivot):
    """
    ROW-REDUCED ECHELON FORM algorithm that takes the pivot element location and stabilizer tableau,
    and converts the elements below the pivot to the standard row echelon form.
    This is one of the steps of the full row reduced echelon form algorithm.

    :param x_matrix: x matrix in the symplectic representation of the stabilizers
    :type x_matrix: np.ndarray
    :param z_matrix: z matrix in the symplectic representation of the stabilizers
    :type z_matrix: np.ndarray
    :param r_vector: the vector of phase factors.
    :type r_vector: np.ndarray
    :param pivot: the location of the pivot element [i,j] on the i-th row and j-th column of the stabilizer tableau.
    :type pivot: list
    :return: updated stabilizer tableau and updated pivot
    :rtype: numpy.ndarray, numpy.ndarray, numpy.ndarray, list
    """

    # pauli_x_list  = list of the rows (generators) with a pauli X operator in the pivot column
    # pauli_y_list  = list of the rows (generators) with a pauli Y operator in the pivot column
    # pauli_z_list  = list of the rows (generators) with a pauli Z operator in the pivot column
    pauli_x_list, pauli_y_list, pauli_z_list = pauli_type_finder(
        x_matrix, z_matrix, pivot
    )
    pauli_list_dict = {"x": pauli_x_list, "y": pauli_y_list, "z": pauli_z_list}
    # case of no pauli operator!
    if not (pauli_x_list or pauli_y_list or pauli_z_list):
        pivot = [pivot[0], pivot[1] + 1]
        return x_matrix, z_matrix, r_vector, pivot

    # case of only 1 kind of pauli
    elif pauli_x_list and (not pauli_y_list) and (not pauli_z_list):  # only X
        return _process_one_pauli(x_matrix, z_matrix, r_vector, pivot, pauli_x_list)

    elif pauli_y_list and (not pauli_x_list) and (not pauli_z_list):  # only Y
        return _process_one_pauli(x_matrix, z_matrix, r_vector, pivot, pauli_y_list)

    elif pauli_z_list and (not pauli_x_list) and (not pauli_y_list):  # only Z
        return _process_one_pauli(x_matrix, z_matrix, r_vector, pivot, pauli_z_list)

    # case of two kinds of pauli
    elif not pauli_x_list:  # pauli y and z exist in the column below pivot
        return _process_two_pauli(
            x_matrix, z_matrix, r_vector, pivot, pauli_list_dict, "y", "z"
        )

    elif not pauli_y_list:  # pauli x and z exist in the column below pivot
        return _process_two_pauli(
            x_matrix, z_matrix, r_vector, pivot, pauli_list_dict, "x", "z"
        )

    elif not pauli_z_list:  # pauli x and y exist in the column below pivot
        return _process_two_pauli(
            x_matrix, z_matrix, r_vector, pivot, pauli_list_dict, "x", "y"
        )

    # case of all three kinds of paulis available in the column
    else:
        old_pivot = [0, 0]
        old_pivot[0] = pivot[0]
        old_pivot[1] = pivot[1]
        x_matrix, z_matrix, r_vector, _ = _process_two_pauli(
            x_matrix, z_matrix, r_vector, old_pivot, pauli_list_dict, "x", "z"
        )
        # update pauli lists
        pauli_x_list, pauli_y_list, pauli_z_list = pauli_type_finder(
            x_matrix, z_matrix, pivot
        )
        for row_k in pauli_y_list:
            # multiplying the pauli Y with pauli X to make it Z
            x_matrix, z_matrix, r_vector = row_sum(
                x_matrix, z_matrix, r_vector, pivot[0], row_k
            )
            # multiplying the now Z row with another Z to eliminate it
            x_matrix, z_matrix, r_vector = row_sum(
                x_matrix, z_matrix, r_vector, pivot[0] + 1, row_k
            )
        pivot = [pivot[0] + 2, pivot[1] + 1]
        return x_matrix, z_matrix, r_vector, pivot


def rref(x_matrix, z_matrix, r_vector):
    """
    Takes stabilizer tableau, and converts it to the standard row echelon form.

    :param x_matrix: x matrix in the symplectic representation of the stabilizers
    :type x_matrix: np.ndarray
    :param z_matrix: z matrix in the symplectic representation of the stabilizers
    :type z_matrix: np.ndarray
    :param r_vector: the vector of phase factors.
    :type r_vector: np.ndarray
    :return: stabilizer tableau in the row reduced echelon form
    :rtype: numpy.ndarray, numpy.ndarray, numpy.ndarray
    """
    # TODO: check the validity of input x and z matrices. Partially done by checking the rank by assertion below.
    pivot = [0, 0]
    n_qubits = np.shape(x_matrix)[0]
    while pivot[0] <= n_qubits - 1 and pivot[1] <= n_qubits - 1:
        x_matrix, z_matrix, r_vector, pivot = one_step_rref(
            x_matrix, z_matrix, r_vector, pivot
        )
    assert (
        pivot[0] >= n_qubits - 1
    ), "Invalid input. One of the stabilizers is identity on all qubits!"  # rank check
    return x_matrix, z_matrix, r_vector
