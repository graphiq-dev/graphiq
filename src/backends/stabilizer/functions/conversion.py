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


def from_graph():
    pass


def from_stabilizer():
    pass


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
