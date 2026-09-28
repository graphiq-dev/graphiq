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
import networkx as nx

import graphiq.backends.density_matrix.functions as dmf
from graphiq.backends.density_matrix import numpy as np
from graphiq.backends.density_matrix.state import DensityMatrix


# @pytest.fixture(params=['numpy_fixture', 'jax_fixture'])
# def library_fixture(request):
#     return request.getfixturevalue(request.param)
# import graphiq.backends.density_matrix
# from graphiq.backends.density_matrix import numpy as np
# import graphiq.backends.density_matrix.functions as dmf


def test_density_matrix_creation_ndararay():
    state_rep = DensityMatrix(np.eye(2))

    assert state_rep.data.shape[0] == state_rep.data.shape[1] == 2
    assert dmf.is_psd(state_rep.data)


def test_density_matrix_creation_graph():
    g = nx.Graph([(1, 2), (1, 3)])
    state_rep = DensityMatrix.from_graph(g)

    assert state_rep.data.shape[0] == state_rep.data.shape[1] == 8
    assert dmf.is_psd(state_rep.data)


def test_density_matrix_get_single_qubit_gate():
    # qubit 0 is the leftmost tensor factor
    assert np.allclose(
        dmf.get_one_qubit_gate(3, 0, dmf.sigmaz()),
        np.diag([1, 1, 1, 1, -1, -1, -1, -1]),
    )
    assert np.allclose(
        dmf.get_two_qubit_controlled_gate(3, 0, 1, dmf.sigmax()),
        np.kron(np.kron(np.diag([1, 0]), np.eye(2)), np.eye(2))
        + np.kron(np.kron(np.diag([0, 1]), dmf.sigmax()), np.eye(2)),
    )


def test_density_matrix_eq_against_a_non_density_matrix():
    dm = DensityMatrix(np.array([[1.0, 0.0], [0.0, 0.0]]))
    assert (dm == 3) is False
    assert (dm == None) is False


def test_density_matrix_eq_different_sizes():
    # a (1, 1) matrix must not broadcast into equality with a (2, 2) one
    assert (DensityMatrix(np.ones((1, 1))) == DensityMatrix(np.ones((2, 2)))) is False
    assert (DensityMatrix(np.eye(2) / 2) == DensityMatrix(np.eye(4) / 4)) is False


def test_forced_outcome_0_with_zero_probability_does_not_divide_by_zero():
    # p(1) is 1 up to rounding and p(0) is exactly 0, so outcome 0 is impossible
    state = DensityMatrix(
        np.diag([0.0, 1.0 - 1e-16]).astype(complex), normalized=False
    )
    projectors = [np.diag([1.0, 0.0]), np.diag([0.0, 1.0])]

    outcome = state.apply_measurement(projectors, measurement_determinism=0)

    assert outcome == 1
    assert not np.any(np.isnan(state.data))
