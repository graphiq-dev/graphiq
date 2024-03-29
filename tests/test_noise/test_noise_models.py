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
import numpy as np

import graphiq.backends.density_matrix.functions as dmf
import graphiq.backends.stabilizer.functions.clifford as sfc
import graphiq.noise.noise_models as nm
from graphiq.backends.stabilizer.state import MixedStabilizer
from graphiq.state import QuantumState


def _state_initialization_dm(n_quantum, state=dmf.state_ketz0()):
    return QuantumState(
        dmf.create_n_product_state(n_quantum, state),
        rep_type="dm",
    )


def _state_initialization_stabilizer(n_quantum):
    return QuantumState(
        sfc.create_n_ket0_state(n_quantum),
        rep_type="s",
    )


def test_depolarizing_noise():
    n_quantum = 3
    qubit_position = 1
    qubit_state = dmf.state_ketx0()
    state1 = _state_initialization_dm(n_quantum, qubit_state)
    state2 = _state_initialization_dm(n_quantum, qubit_state)
    depolarizing_prob = 0.1
    noise1 = nm.DepolarizingNoise(depolarizing_prob)

    kraus_x = np.sqrt(depolarizing_prob / 3) * dmf.get_one_qubit_gate(
        n_quantum, qubit_position, dmf.sigmax()
    )
    kraus_y = np.sqrt(depolarizing_prob / 3) * dmf.get_one_qubit_gate(
        n_quantum, qubit_position, dmf.sigmay()
    )
    kraus_z = np.sqrt(depolarizing_prob / 3) * dmf.get_one_qubit_gate(
        n_quantum, qubit_position, dmf.sigmaz()
    )

    kraus_ops = [
        np.sqrt(1 - depolarizing_prob) * np.eye(2**n_quantum),
        kraus_x,
        kraus_y,
        kraus_z,
    ]

    noise1.apply(state1, n_quantum, [qubit_position])
    state2.rep_data.apply_channel(kraus_ops)

    assert np.allclose(state1.rep_data.data, state2.rep_data.data)


def test_no_noise():
    n_quantum = 5
    qubit_state = dmf.state_kety0()
    state1 = _state_initialization_dm(n_quantum, qubit_state)
    state2 = _state_initialization_dm(n_quantum, qubit_state)
    noise = nm.NoNoise()
    noise.apply(state1)

    assert np.allclose(state1.rep_data.data, state2.rep_data.data)


def test_one_qubit_replacement():
    n_quantum = 5
    qubit_position = 2
    state1 = _state_initialization_dm(n_quantum, dmf.state_ketx0())
    state2 = _state_initialization_dm(n_quantum, dmf.state_ketx0())
    noise = nm.OneQubitGateReplacement(
        dmf.parameterized_one_qubit_unitary(np.pi / 2, 0, np.pi)
    )

    hadamard_gate = dmf.get_one_qubit_gate(n_quantum, qubit_position, dmf.hadamard())

    noise.apply(state1, n_quantum, [qubit_position])

    state2.rep_data.apply_unitary(hadamard_gate)
    assert np.allclose(state1.rep_data.data, state2.rep_data.data)


def test_two_qubit_replacement():
    n_quantum = 5
    control_qubit = 2
    target_qubit = 0
    state1 = _state_initialization_dm(n_quantum, dmf.state_ketx1())
    state2 = _state_initialization_dm(n_quantum, dmf.state_ketx1())
    noise = nm.TwoQubitControlledGateReplacement(
        dmf.parameterized_one_qubit_unitary(np.pi, 0, np.pi), phase_factor=0
    )

    cnot_gate = dmf.get_two_qubit_controlled_gate(
        n_quantum, control_qubit, target_qubit, dmf.sigmax()
    )

    noise.apply(state1, n_quantum, control_qubit, target_qubit)

    state2.rep_data.apply_unitary(cnot_gate)
    assert np.allclose(state1.rep_data.data, state2.rep_data.data)


def test_pauli_error():
    n_quantum = 5
    qubit_position = 2
    state1 = _state_initialization_dm(n_quantum, dmf.state_ketx0())
    state2 = _state_initialization_dm(n_quantum, dmf.state_ketx0())
    noise = nm.PauliError("X")

    x_gate = dmf.get_one_qubit_gate(n_quantum, qubit_position, dmf.sigmax())
    noise.apply(state1, n_quantum, [qubit_position])

    state2.rep_data.apply_unitary(x_gate)
    assert np.allclose(state1.rep_data.data, state2.rep_data.data)


def test_photon_loss_stabilizer():
    n_quantum = 2
    state = _state_initialization_stabilizer(n_quantum)
    noise = nm.PhotonLoss(0.1)
    noise.apply(state, n_quantum, 1)
    assert isinstance(state.rep_data, MixedStabilizer)
    assert state.rep_data.probability == 0.9


def test_photon_loss_dm():
    n_quantum = 2
    state = _state_initialization_dm(n_quantum)
    noise = nm.PhotonLoss(0.1)
    noise.apply(state, n_quantum, 1)
    assert state.rep_data.trace == 0.9
    assert not state.rep_data.normalized
