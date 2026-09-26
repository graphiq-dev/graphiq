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
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pytest

import graphiq.backends.density_matrix.functions as dmf
import graphiq.backends.state_rep_conversion as rc
from graphiq.backends.density_matrix.state import DensityMatrix


def test_y_measurement():
    graph1 = nx.Graph([(1, 2), (2, 3), (3, 4), (1, 3)])
    rho = DensityMatrix(rc.graph_to_density(graph1))
    kety0 = dmf.state_kety0()
    kety1 = dmf.state_kety1()
    projector0 = dmf.get_one_qubit_gate(4, 2, kety0 @ np.conjugate(kety0.T))
    projector1 = dmf.get_one_qubit_gate(4, 2, kety1 @ np.conjugate(kety1.T))
    projectors = [projector0, projector1]
    rho.apply_measurement(projectors, measurement_determinism=0)
    graph_adj = rc.density_to_graph(rho.data, validate=False)
    nx.draw(nx.from_numpy_array(graph_adj), with_labels=True)
    plt.show()


def test_x_measurement():
    graph1 = nx.Graph([(1, 2), (2, 3), (3, 4), (1, 3), (1, 4), (2, 4), (1, 5), (2, 5)])
    # nx.draw(graph1, with_labels=True)
    rho = DensityMatrix(rc.graph_to_density(graph1))
    ketx0 = dmf.state_ketx0()
    ketx1 = dmf.state_ketx1()
    graph2 = nx.Graph([(1, 2), (1, 4), (2, 4), (1, 5), (2, 5)])
    graph2.add_node(3)
    nx.draw(graph2, with_labels=True)
    rho2 = DensityMatrix(rc.graph_to_density(graph2))
    projector0 = dmf.get_one_qubit_gate(5, 2, ketx0 @ np.conjugate(ketx0.T))
    projector1 = dmf.get_one_qubit_gate(5, 2, ketx1 @ np.conjugate(ketx1.T))
    projectors = [projector0, projector1]
    rho.apply_measurement(projectors, measurement_determinism=1)
    print(f"rho is {rho.data}")
    print(f"rho2 is {rho2.data}")

    graph_adj = rc.density_to_graph(rho.data, validate=False)
    # nx.draw(nx.from_numpy_array(graph_adj),with_labels=True)
    plt.show()


_PLUS = np.full((2, 2), 0.5)
_Z_PROJECTORS = [np.diag([1.0, 0.0]), np.diag([0.0, 1.0])]


@pytest.mark.parametrize("determinism", [0, 1, "probabilistic"])
def test_measurement_keeps_the_trace_of_a_subnormalised_state(determinism):
    # photon loss leaves a trace below 1, which the measurement must keep
    np.random.seed(0)
    rho = DensityMatrix(0.9 * _PLUS, normalized=False)

    outcome = rho.apply_measurement(_Z_PROJECTORS, measurement_determinism=determinism)

    assert np.allclose(rho.data, 0.9 * _Z_PROJECTORS[outcome])


@pytest.mark.parametrize("determinism", [0, 1])
def test_measurement_keeps_a_trace_just_below_one(determinism):
    # a loss of 1e-6 is far above rounding drift, so it is kept too
    rho = DensityMatrix(0.999999 * _PLUS, normalized=False)

    rho.apply_measurement(_Z_PROJECTORS, measurement_determinism=determinism)

    assert np.real(np.trace(rho.data)) == pytest.approx(0.999999, rel=1e-12)


def _path3_with_trace_one_up_to_rounding():
    # four gate and dephasing steps leave the trace at 1 - 1.4e-15
    h = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    unitaries = [np.kron(np.eye(2), np.kron(h, h)), np.kron(h, np.eye(4))]
    dephasing = [np.sqrt(0.9) * np.eye(8), np.sqrt(0.1) * np.diag([1, -1] * 4)]
    rho = DensityMatrix(rc.graph_to_density(nx.path_graph(3)))
    for step in range(4):
        rho.apply_unitary(unitaries[step % 2])
        rho.apply_channel(dephasing)
    return rho


@pytest.mark.parametrize("determinism", [0, 1])
def test_measurement_of_a_state_with_trace_one_up_to_rounding_is_unchanged(
    determinism,
):
    rho = _path3_with_trace_one_up_to_rounding()
    before = rho.data.copy()
    trace = np.real(np.trace(before))
    assert np.trace(before) != 1.0
    projectors = [
        np.kron(np.diag([1.0, 0.0]), np.eye(4)),
        np.kron(np.diag([0.0, 1.0]), np.eye(4)),
    ]

    outcome = rho.apply_measurement(projectors, measurement_determinism=determinism)

    m = projectors[outcome]
    expected = (m @ before @ np.transpose(np.conjugate(m))) / np.real(
        np.trace(before @ m)
    )
    # scaling by this trace would change the matrix, so the check can tell
    assert not np.array_equal(trace * expected, expected)
    assert np.array_equal(rho.data, expected)
