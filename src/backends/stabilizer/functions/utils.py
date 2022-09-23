"""
Miscellaneous functions
"""

import numpy as np

from src.backends.density_matrix import functions as dmf


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
        generator = n_column * [""]
        for j in range(n_column):
            if x_matrix[i, j] == 1 and z_matrix[i, j] == 0:
                generator[j] = "X"
            elif x_matrix[i, j] == 1 and z_matrix[i, j] == 1:
                generator[j] = "Y"
            elif x_matrix[i, j] == 0 and z_matrix[i, j] == 1:
                generator[j] = "Z"
            else:
                generator[j] = "I"
        generator_list.append("".join(generator))
    return generator_list


def string_to_symplectic(generator_list):
    """
    Convert a string list representation of stabilizer generators to a symplectic representation

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


def is_symplectic(input_matrix):
    """
    Check if a given matrix is symplectic.

    :param input_matrix: a binary matrix
    :type input_matrix: numpy.ndarray
    :return: True if the input matrix is symplectic; False otherwise
    :rtype: bool
    """
    dim = int(input_matrix.shape[1] / 2)
    symplectic_p = np.block(
        [[np.zeros((dim, dim)), np.eye(dim)], [np.eye(dim), np.zeros((dim, dim))]]
    ).astype(int)

    return np.array_equal(
        binary_symplectic_product(input_matrix, input_matrix), symplectic_p
    )


def is_symplectic_self_orthogonal(input_matrix):
    """
    Check if a given symplectic matrix is self-orthogonal.

    :param input_matrix: a binary symplectic matrix
    :type input_matrix: numpy.ndarray
    :return: True if the input matrix is self-orthogonal; False otherwise
    :rtype: bool
    """
    dim = int(input_matrix.shape[1] / 2)

    return np.array_equal(
        binary_symplectic_product(input_matrix, input_matrix), np.zeros((dim, dim))
    )


def binary_symplectic_product(matrix1, matrix2):
    r"""
    Compute the binary symplectic product of two matrices matrix1 (:math:`M_1`) and matrix2 (:math:`M_2`)

    The symplectic inner product between :math:`M_1` and :math:`M_2` is :math:`M_1 P M_2^T`,
    where :math:`P = \\begin{bmatrix} 0 & I \\\ I & 0 \\end{bmatrix}`.

    :param matrix1: a binary symplectic matrix
    :type matrix1: numpy.ndarray
    :param matrix2: a binary symplectic matrix
    :type matrix2: numpy.ndarray
    :return: the symplectic product of these two matrices
    :rtype: int
    """
    assert matrix1.shape[0] == matrix2.shape[0]
    dim = matrix1.shape[0]
    symplectic_p = np.block(
        [[np.zeros((dim, dim)), np.eye(dim)], [np.eye(dim), np.zeros((dim, dim))]]
    ).astype(int)
    return ((matrix1 @ symplectic_p @ matrix2.T) % 2).astype(int)
