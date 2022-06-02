import numpy as np


def symplectic_to_string(x_matrix, z_matrix):
    """
    Convert a binary symplectic representation to a list of strings

    :param x_matrix: X part of the binary symplectic representation
    :param z_matrix: Z part of the binary symplectic representation
    :return: a list of strings that represent stabilizer generators
    :rtype: string list
    """
    assert x_matrix.shape == z_matrix.shape
    n_row, n_column = x_matrix.shape
    generator_list = []
    for i in range(n_row):
        generator = ''
        for j in range(n_column):
            if x_matrix[i, j] == 1 and z_matrix[i, j] == 0:
                generator = generator + 'X'
            elif x_matrix[i, j] == 1 and z_matrix[i, j] == 1:
                generator = generator + 'Y'
            elif x_matrix[i, j] == 0 and z_matrix[i, j] == 1:
                generator = generator + 'Z'
            else:
                generator = generator + 'I'
        generator_list.append(generator)
    return generator_list


def string_to_symplectic(generator_list):
    """

    :param generator_list: a list of strings
    :return: two binary matrices, one for X part, the other for Z part
    """
    n_row = len(generator_list)
    n_column = len(generator_list[0])
    x_matrix = np.zeros((n_row, n_column))
    z_matrix = np.zeros((n_row, n_column))
    for i in range(n_row):
        generator = generator_list[i]
        for j in range(n_column):
            if generator[j].lower() == 'x':
                x_matrix[i, j] = 1
            elif generator[j].lower() == 'y':
                x_matrix[i, j] = 1
                z_matrix[i, j] = 1
            elif generator[j].lower() == 'z':
                z_matrix[i, j] = 1
    return x_matrix, z_matrix


def row_swap(x_matrix, i, j):
    """
    Swap two rows of a matrix

    :param x_matrix:
    """
    x_matrix[[i, j]] = x_matrix[[j, i]]
    return x_matrix


def add_rows(x_matrix, i, j):
    """
    Add two rows together modulo 2 and put it in the row of the second input
    """
    x_matrix[j] = (x_matrix[i] + x_matrix[j]) % 2
    return x_matrix


def hadamard_transform(x_matrix, z_matrix, positions):
    """
    Apply Hadamard gate on each qubit specified by
    Column swap between X and Z
    """
    temp1 = list(z_matrix[:, positions])
    temp2 = list(x_matrix[:, positions])
    z_matrix[:, positions] = temp2
    x_matrix[:, positions] = temp1
    return x_matrix, z_matrix


def row_reduction(x_matrix, z_matrix, pivot):
    """
    Returns the row reduced matrix X, the transformed matrix Z and the (rank-1) of the X matrix

    :param x_matrix:
    :param z_matrix:
    :param pivot:
    :return:
    """
    n, m = np.shape(x_matrix)
    rank = 0
    if pivot[1] == (m - 1):
        return x_matrix, z_matrix, pivot[0]
    else:
        # list of rows with value 1 under the pivot element
        the_ones = []
        for a in range(pivot[0], n):
            if x_matrix[a, pivot[1]] == 1:
                the_ones.append(a)
        # check if the column below is empty to skip it
        if not the_ones:
            pivot = [pivot[0], pivot[1] + 1]
            x_matrix, z_matrix, rank = row_reduction(x_matrix, z_matrix, pivot)
        else:
            x_matrix = row_swap(x_matrix, the_ones[0], pivot[0])
            z_matrix = row_swap(z_matrix, the_ones[0], pivot[0])
            the_ones.remove(the_ones[0])
            for b in the_ones:
                x_matrix = add_rows(x_matrix, pivot[0], b)
                z_matrix = add_rows(z_matrix, pivot[0], b)
            pivot = [pivot[0] + 1, pivot[1] + 1]
            x_matrix, z_matrix, rank = row_reduction(x_matrix, z_matrix, pivot)
    return x_matrix, z_matrix, pivot[0]
