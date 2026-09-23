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
import pytest

from graphiq.backends.stabilizer.state import *


def test_state1():
    state = Stabilizer(4)
    print(state.tableau.destabilizer_to_labels())
    print(state.tableau.stabilizer_to_labels())
    state.tableau.stabilizer_z = np.array(
        [[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]]
    )
    state.tableau.destabilizer_z = np.array(
        [[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]]
    )
    state.apply_hadamard(1)
    state.tableau.destabilizer_from_labels(["XYXI", "ZYZI", "YIII", "ZZZZ"])
    print(state.tableau.destabilizer)


def _plus_on_qubit_1(n_qubits):
    state = Stabilizer(n_qubits)
    state.apply_hadamard(1)
    return state


def test_trace_out_qubits_discards_the_given_qubits():
    # |0>|+>|0>: tracing out qubit 0 must leave |+>|0>, not keep qubit 0 alone
    state = _plus_on_qubit_1(3)
    state.trace_out_qubits([0], measurement_determinism=0)

    expected = Stabilizer(2)
    expected.apply_hadamard(0)
    assert state.n_qubits == 2
    assert state == expected


def test_mixed_trace_out_qubits_discards_the_given_qubits():
    state = MixedStabilizer(
        [(0.5, _plus_on_qubit_1(3).tableau), (0.5, _plus_on_qubit_1(3).tableau)]
    )
    state.trace_out_qubits([0], measurement_determinism=0)

    expected = Stabilizer(2)
    expected.apply_hadamard(0)
    assert state.n_qubits == 2
    for _, tableau in state.mixture:
        assert Stabilizer(tableau) == expected


def test_mixed_stabilizer_tableau_raises():
    state = MixedStabilizer([(1.0, Stabilizer(2).tableau)])
    with pytest.raises(TypeError):
        state.tableau
