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
Miscellaneous functions
"""

import numpy as np

from graphiq.backends.density_matrix import functions as dmf


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
    return np.array_equal(
        binary_symplectic_product(input_matrix, input_matrix),
        np.zeros((input_matrix.shape[0],) * 2),
    )


def is_stabilizer(table):
    """
    Check if a given n x 2n binary symplectic table is a valid set of stabilizer generators,
    i.e. its rows pairwise commute and are linearly independent over GF(2).

    :param table: an n x 2n binary symplectic table
    :type table: numpy.ndarray
    :return: True if the rows pairwise commute and are linearly independent over GF(2); False
        otherwise
    :rtype: bool
    """
    n_rows, n_columns = table.shape
    assert n_columns == 2 * n_rows
    if not is_symplectic_self_orthogonal(table):
        return False
    # GF(2) rank by Gaussian elimination; a real-valued rank would not do, e.g.
    # ZZI, IZZ, ZIZ has GF(2) rank 2 but real rank 3
    reduced = np.array(table, dtype=int) % 2
    rank = 0
    for column in range(n_columns):
        pivots = np.nonzero(reduced[rank:, column])[0]
        if len(pivots) == 0:
            continue
        pivot = rank + pivots[0]
        reduced[[rank, pivot]] = reduced[[pivot, rank]]
        for row in range(n_rows):
            if row != rank and reduced[row, column]:
                reduced[row] = (reduced[row] + reduced[rank]) % 2
        rank += 1
        if rank == n_rows:
            break
    return rank == n_rows


def binary_symplectic_product(matrix1, matrix2):
    """
    Compute the binary symplectic product of two matrices matrix1 (:math:`M_1`) and matrix2 (:math:`M_2`)

    The symplectic inner product between :math:`M_1` and :math:`M_2` is :math:`M_1 P M_2^T`,
    where :math:`P = \\begin{bmatrix} 0 & I \\\\ I & 0 \\end{bmatrix}` is sized from the shared
    number of columns (2n), not the row count -- the two matrices need not have the same number of
    rows (e.g. a stabilizer's n rows against a Clifford tableau's 2n).

    :param matrix1: a binary symplectic matrix
    :type matrix1: numpy.ndarray
    :param matrix2: a binary symplectic matrix
    :type matrix2: numpy.ndarray
    :return: a (number of rows in matrix1, number of rows in matrix2) binary matrix, one
        entry per row pair
    :rtype: numpy.ndarray
    """
    assert matrix1.shape[1] == matrix2.shape[1]
    dim = matrix1.shape[1] // 2
    symplectic_p = np.block(
        [[np.zeros((dim, dim)), np.eye(dim)], [np.eye(dim), np.zeros((dim, dim))]]
    ).astype(int)
    return ((matrix1 @ symplectic_p @ matrix2.T) % 2).astype(int)
