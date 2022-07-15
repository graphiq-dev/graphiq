import pytest
import numpy as np
from functools import reduce
import src.noise.noise_models as nm
from src.backends.density_matrix.compiler import DensityMatrixCompiler
import src.ops as ops
from src.backends.density_matrix.state import DensityMatrix


def test_multi_qubit_depolarizing():
    n_quantum = 3
    state1 = DensityMatrix(n_quantum)
    state2 = DensityMatrix(n_quantum)

    noise1 = nm.MultiQubitDepolarizingNoise(0.1)
    noise2 = nm.OneQubitDepolarizingNoise(0.1)
    kraus_op_list1 = noise1.get_backend_dependent_noise(state1, n_quantum, [1])
    kraus_op_list2 = noise2.get_backend_dependent_noise(state1, n_quantum, [1])
    noise1.apply(state1, n_quantum, [1])
    noise2.apply(state2, n_quantum, [1])

    assert np.allclose(state1.data, state2.data)

    for i in range(len(kraus_op_list1)):
        assert np.allclose(kraus_op_list1[i], kraus_op_list2[i])
