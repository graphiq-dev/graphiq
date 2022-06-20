import numpy as np
import src.backends.stabilizer.functions as sf
from functools import reduce


def test_symplectic_to_string():
    x_matrix1 = np.zeros((4, 4))
    z_matrix1 = np.eye(4)
    expected1 = ['ZIII', 'IZII', 'IIZI', 'IIIZ']
    assert reduce(lambda x, y: x and y,
                  map(lambda a, b: a == b, sf.symplectic_to_string(x_matrix1, z_matrix1), expected1),
                  True)

    x_matrix2 = np.array([[1, 0, 0, 0], [0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
    z_matrix2 = np.eye(4)
    expected2 = ['YIII', 'IZII', 'IXZI', 'IIXZ']
    assert reduce(lambda x, y: x and y,
                  map(lambda a, b: a == b, sf.symplectic_to_string(x_matrix2, z_matrix2), expected2),
                  True)


def test_string_to_symplectic():
    generator_list = ['ZIII', 'IZII', 'IIZI', 'IIIZ']
    x_matrix, z_matrix = sf.string_to_symplectic(generator_list)
    assert np.array_equal(x_matrix, np.zeros((4, 4)))
    assert np.array_equal(z_matrix, np.eye(4))
