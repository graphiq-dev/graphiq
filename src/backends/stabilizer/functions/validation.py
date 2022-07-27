import numpy as np


def is_symplectic(input_matrix):
    """
    Check if a given matrix is symplectic.

    :param input_matrix:
    :type input_matrix:
    :return:
    :rtype: bool
    """
    dim = int(input_matrix.shape[1] / 2)
    symplectic_p = np.block(
        [[np.zeros((dim, dim)), np.eye(dim)], [np.eye(dim), np.zeros((dim, dim))]]
    ).astype(int)

    return np.array_equal(binary_symplectic_product(input_matrix, input_matrix), symplectic_p)


def is_symplectic_self_orthogonal(input_matrix):
    """
    Check if a given symplectic matrix is self-orthogonal.

    :param input_matrix:
    :type input_matrix:
    :return:
    :rtype: bool
    """
    dim = int(input_matrix.shape[1] / 2)

    return np.array_equal(
        binary_symplectic_product(input_matrix, input_matrix), np.zeros((dim, dim))
    )


def binary_symplectic_product(matrix1, matrix2):
    """
    Compute the binary symplectic product of two matrices matrix1 (:math:`M_1`) and matrix2 (:math:`M_2`)

    The symplectic inner product between :math:`M_1` and :math:`M_2` is :math:`M_1 P M_2^T`,
    where :math:`P = \\begin{bmatrix} 0 & I \\\ I & 0 \\end{bmatrix}`.


    :param matrix1:
    :type matrix1:
    :param matrix2:
    :type matrix2:
    :return:
    :rtype:
    """
    assert matrix1.shape[0] == matrix2.shape[0]
    dim = matrix1.shape[0]
    symplectic_p = np.block(
        [[np.zeros((dim, dim)), np.eye(dim)], [np.eye(dim), np.zeros((dim, dim))]]
    ).astype(int)
    return ((matrix1 @ symplectic_p @ matrix2.T) % 2).astype(int)


def is_pure():
    pass


def is_graph():
    pass


def is_stabilizer():
    pass
