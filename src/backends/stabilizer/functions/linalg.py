"""
Functions that are binary matrix linear algebra
"""

import numpy as np


def row_swap(input_matrix, first_row, second_row):
    """
    Swap two rows of a matrix

    :param input_matrix: a binary matrix
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


def add_rows(input_matrix, row_to_add, target_row):
    """
    Add two rows together modulo 2 and put it in the row of the second input

    :param input_matrix: a binary matrix
    :type input_matrix: numpy.ndarray
    :param row_to_add: the index of the row to add
    :type row_to_add: int
    :param target_row: the index of the row where the result is put
    :type target_row: int
    :return: the matrix after adding two rows modulo 2 and putting in the row of the second input
    :rtype: numpy.ndarray
    """
    tmp = (input_matrix[row_to_add] + input_matrix[target_row]) % 2
    input_matrix[target_row] = tmp.astype(int)
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


def add_columns(input_matrix, col_to_add, target_col):
    """
    Add two rows together modulo 2 and put it in the row of the second input

    :param input_matrix: a binary matrix
    :type input_matrix: numpy.ndarray
    :param col_to_add: the index of the column to add
    :type col_to_add: int
    :param target_col: the index of the column where the result is put
    :type target_col: int
    :return: the matrix after adding two column modulo 2 and putting in the column of the second input
    :rtype: numpy.ndarray
    """
    tmp = (input_matrix[:, col_to_add] + input_matrix[:, target_col]) % 2
    input_matrix[:, target_col] = tmp.astype(int)
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
    n_rows1, n_columns1 = matrix_one.shape
    n_rows2, n_columns2 = matrix_two.shape
    assert n_rows1 == n_rows2

    assert (
        first_col < n_columns1 and second_col < n_columns2
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
    temp1 = z_matrix[:, positions]
    temp2 = x_matrix[:, positions]
    z_matrix[:, positions] = temp2
    x_matrix[:, positions] = temp1
    return x_matrix, z_matrix


def row_reduction(x_matrix, z_matrix):
    """
    Turn the x_matrix into a row reduced echelon form. Apply the same row operations on z_matrix.

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
        # list of rows with value 1 under the pivot element
        the_ones = [i for i in range(pivot[0], n_row) if x_matrix[i, pivot[1]] == 1]
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
        the_ones = [i for i in range(pivot[0], n_row) if x_matrix[i, pivot[1]] == 1]
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


def g_function(x1, z1, x2, z2):
    """
    A helper function to use in rowsum function. Takes 4 bits (2 Pauli matrices in the binary representation) as input
    and returns the phase factor needed when the two Pauli matrices are multiplied: Pauli_1 * Pauli_2
    As a convention for the phase, each Y operator introduces -i phase factor.

    Refer to section III of arXiv:quant-ph/0406196v5

    :param x1: the x bit of the first Pauli operator
    :type x1: int
    :param z1: the z bit of the first Pauli operator
    :type z1: int
    :param x2: the x bit of the second Pauli operator
    :type x2: int
    :param z2: the z bit of the second Pauli operator
    :type z2: int
    :return: the exponent k in the phase factor: i^k where "i" is the unit imaginary number
    :rtype: int
    """

    if x1 == z1 == 0:  # both equal to zero
        return 0
    elif x1 == z1 == 1:
        return z2 - x2
    elif x1 == 1 and z1 == 0:
        return z2 * (2 * x2 - 1)
    else:
        return x2 * (1 - 2 * z2)


def row_sum(x_matrix, z_matrix, r_vector, iphase_vector, row_to_add, target_row):
    """
    Takes the full stabilizer tableau as input and sets the stabilizer generator in the target_row equal to
    (row_to_add + target_row) while keeping track of the phase vectors.
    This is based on the section III of the article arXiv:quant-ph/0406196v5

    :param x_matrix: x matrix in the symplectic representation of the stabilizers
    :type x_matrix: np.ndarray
    :param z_matrix: z matrix in the symplectic representation of the stabilizers
    :type z_matrix: np.ndarray
    :param r_vector: the vector of phase factors for +1 or -1 phase.
    :type r_vector: np.ndarray
    :param iphase_vector: the vector of phase factors for +i or -i phase.
    :type iphase_vector: np.ndarray
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

    phases = 2 * r_vector[target_row] + iphase_vector[target_row]
    phases += 2 * r_vector[row_to_add] + iphase_vector[row_to_add] + g_sum
    phases = phases % 4

    # for exponent of (-1)
    r_vector[target_row] = int(phases / 2)
    # for exponent of i
    iphase_vector[target_row] = phases % 2

    # calculating the resulting new matrices after adding row i to h.
    x_matrix = add_rows(x_matrix, row_to_add, target_row)
    z_matrix = add_rows(z_matrix, row_to_add, target_row)

    return x_matrix, z_matrix, r_vector, iphase_vector
