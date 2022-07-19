import numpy as np
import src.backends.density_matrix.functions as dmf


def symplectic_to_string(x_matrix, z_matrix):
    """
    Convert a binary symplectic representation to a list of strings

    :param x_matrix: X part of the binary symplectic representation
    :param z_matrix: Z part of the binary symplectic representation
    :return: a list of strings that represent stabilizer generators
    :rtype: list[str]
    """
    assert x_matrix.shape == z_matrix.shape
    n_row, n_column = x_matrix.shape
    generator_list = []
    for i in range(n_row):
        generator = ""
        for j in range(n_column):
            if x_matrix[i, j] == 1 and z_matrix[i, j] == 0:
                generator = generator + "X"
            elif x_matrix[i, j] == 1 and z_matrix[i, j] == 1:
                generator = generator + "Y"
            elif x_matrix[i, j] == 0 and z_matrix[i, j] == 1:
                generator = generator + "Z"
            else:
                generator = generator + "I"
        generator_list.append(generator)
    return generator_list


def string_to_symplectic(generator_list):
    """
    Convert a string list representation of stabilizer generators to a symplectic representaation

    :param generator_list: a list of strings
    :type generator_list: list[str]
    :return: two binary matrices, one for X part, the other for Z part
    :rtype: numpy.ndarray, numpy.ndarray
    """
    n_row = len(generator_list)
    n_column = len(generator_list[0])
    x_matrix = np.zeros((n_row, n_column))
    z_matrix = np.zeros((n_row, n_column))
    for i in range(n_row):
        generator = generator_list[i]
        for j in range(n_column):
            if generator[j].lower() == "x":
                x_matrix[i, j] = 1
            elif generator[j].lower() == "y":
                x_matrix[i, j] = 1
                z_matrix[i, j] = 1
            elif generator[j].lower() == "z":
                z_matrix[i, j] = 1
    return x_matrix, z_matrix


def row_swap(input_matrix, first_row, second_row):
    """
    Swap two rows of a matrix

    :param input_matrix: a matrix
    :type input_matrix: numpy.ndarray
    :param first_row: the first row
    :type first_row: int
    :param second_row: the second row
    :type second_row: int
    :return: the matrix after swaping those two row
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


def hadamard_transform(x_matrix, z_matrix, positions):
    """
    Apply Hadamard gate on each qubit specified by
    Column swap between X and Z

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


def _row_reduction_old(x_matrix, z_matrix, pivot):
    """
    Returns the row reduced matrix X, the transformed matrix Z and the (rank-1) of the X matrix

    :param x_matrix: The X part of the symplectic representation
    :type x_matrix: numpy.ndarray
    :param z_matrix: The Z part of the symplectic representation
    :type z_matrix: numpy.ndarray
    :param pivot: the row, column position for pivoting
    :type pivot: list[int]
    :return: x_matrix, z_matrix, and rank
    """
    n_row, n_column = np.shape(x_matrix)
    rank = 0

    if pivot[1] == (n_column - 1):
        return x_matrix, z_matrix, pivot[0]
    else:
        # list of rows with value 1 under the pivot element
        the_ones = []
        for i in range(pivot[0], n_row):
            if x_matrix[i, pivot[1]] == 1:
                the_ones.append(i)
        # check if the column below is empty to skip it
        if not the_ones:
            pivot = [pivot[0], pivot[1] + 1]
            x_matrix, z_matrix, rank = _row_reduction_old(x_matrix, z_matrix, pivot)
        else:
            x_matrix = row_swap(x_matrix, the_ones[0], pivot[0])
            z_matrix = row_swap(z_matrix, the_ones[0], pivot[0])
            the_ones.remove(the_ones[0])
            for j in the_ones:
                x_matrix = add_rows(x_matrix, pivot[0], j)
                z_matrix = add_rows(z_matrix, pivot[0], j)
            pivot = [pivot[0] + 1, pivot[1] + 1]
            x_matrix, z_matrix, rank = _row_reduction_old(x_matrix, z_matrix, pivot)
    return x_matrix, z_matrix, rank


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


def _row_red_one_step(
    x_matrix, z_matrix, pivot
):  # one step of the algorithm, only on the pivot provided here
    """
    A helper function to apply one step of the row reduction algorithm. It is used in the main row reduction function.

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
        if not the_ones:  # empty under (and including) pivot element on last column
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


def get_stabilizer_element_by_string(generator):
    """
    Return the corresponding tensor of Pauli matrices for the stabilizer generator specified by the input string

    :param generator: a string for one stabilizer element
    :type generator: str
    :return: a matrix representation of the stabilizer element
    :rtype: numpy.ndarray
    """
    stabilizer_elem = 1
    for pauli in generator:
        if pauli.lower() == "x":
            stabilizer_elem = dmf.tensor([stabilizer_elem, dmf.sigmax()])
        elif pauli.lower() == "y":
            stabilizer_elem = dmf.tensor([stabilizer_elem, dmf.sigmay()])
        elif pauli.lower() == "z":
            stabilizer_elem = dmf.tensor([stabilizer_elem, dmf.sigmaz()])
        else:
            stabilizer_elem = dmf.tensor([stabilizer_elem, np.eye(2)])

    return stabilizer_elem
