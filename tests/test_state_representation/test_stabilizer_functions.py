import numpy as np
import src.backends.stabilizer.functions.conversion as sfc
import src.backends.stabilizer.functions.validation as sfv
from functools import reduce


def test_symplectic_to_string():
    x_matrix1 = np.zeros((4, 4))
    z_matrix1 = np.eye(4)
    expected1 = ["ZIII", "IZII", "IIZI", "IIIZ"]
    assert reduce(
        lambda x, y: x and y,
        map(
            lambda a, b: a == b,
            sfc.symplectic_to_string(x_matrix1, z_matrix1),
            expected1,
        ),
        True,
    )

    x_matrix2 = np.array([[1, 0, 0, 0], [0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
    z_matrix2 = np.eye(4)
    expected2 = ["YIII", "IZII", "IXZI", "IIXZ"]
    assert reduce(
        lambda x, y: x and y,
        map(
            lambda a, b: a == b,
            sfc.symplectic_to_string(x_matrix2, z_matrix2),
            expected2,
        ),
        True,
    )


def test_string_to_symplectic():
    generator_list = ["ZIII", "IZII", "IIZI", "IIIZ"]
    x_matrix, z_matrix = sfc.string_to_symplectic(generator_list)
    assert np.array_equal(x_matrix, np.zeros((4, 4)))
    assert np.array_equal(z_matrix, np.eye(4))


def test_symplectic_product():
    adj_matrix = np.array(
        [[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]]
    ).astype(int)
    tableau = np.block([[np.eye(4), adj_matrix]])
    dim = 2

    assert np.array_equal(
        sfv.binary_symplectic_product(tableau, tableau), np.zeros((4, 4))
    )
    assert sfv.is_symplectic_self_orthogonal(tableau)
