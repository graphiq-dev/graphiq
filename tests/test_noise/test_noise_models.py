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
import pytest

import graphiq.backends.density_matrix.functions as dmf
import graphiq.backends.stabilizer.functions.clifford as sfc
import graphiq.noise.noise_models as nm
from graphiq.backends.stabilizer.state import MixedStabilizer, Stabilizer
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


def _two_branch_mixture(n_quantum):
    return QuantumState(
        [
            (0.25, sfc.create_n_ket0_state(n_quantum)),
            (0.75, sfc.create_n_ket0_state(n_quantum)),
        ],
        rep_type="s",
        mixed=True,
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


@pytest.mark.parametrize(
    "state_factory",
    [_state_initialization_stabilizer, _two_branch_mixture],
    ids=["pure", "mixture"],
)
def test_two_qubit_replacement_refuses_the_stabilizer_backend(state_factory):
    # MixedStabilizer is a sibling of Stabilizer, not a subclass, so a mixture
    # must be named explicitly to get the same NotImplementedError as a pure
    # tableau instead of the generic "Backend type is not supported." TypeError.
    n_quantum = 2
    state = state_factory(n_quantum)
    noise = nm.TwoQubitControlledGateReplacement(np.eye(2))

    with pytest.raises(NotImplementedError):
        noise.apply(state, n_quantum, 0, 1)


@pytest.mark.parametrize(
    "state_factory",
    [_state_initialization_stabilizer, _two_branch_mixture],
    ids=["pure", "mixture"],
)
@pytest.mark.parametrize(
    "noise",
    [
        nm.OneQubitGateReplacement(np.eye(2)),
        nm.LocalCliffordError(["sigmax"]),
        nm.AmplitudeDampingNoise(0.1),
        nm.MixedUnitaryError([np.eye(2)], [1.0]),
        nm.CoherentUnitaryError(np.eye(2)),
        nm.GeneralKrausError([np.eye(2)]),
        nm.MeasurementError(np.eye(2)),
        nm.ResetError(),
    ],
    ids=[
        "one_qubit_replacement",
        "local_clifford",
        "amplitude_damping",
        "mixed_unitary",
        "coherent_unitary",
        "general_kraus",
        "measurement",
        "reset",
    ],
)
def test_one_register_noise_refuses_the_stabilizer_backend(noise, state_factory):
    # None of these is representable on the stabilizer backend; each must
    # refuse a mixture the same way it refuses a pure tableau.
    n_quantum = 2
    state = state_factory(n_quantum)

    with pytest.raises(NotImplementedError):
        noise.apply(state, n_quantum, [0])


@pytest.mark.parametrize("depolarizing_prob", [0.0, 0.2, 0.75])
def test_depolarizing_noise_on_a_pure_stabilizer(depolarizing_prob):
    # A pure Stabilizer (what Monte Carlo noise simulation compiles to) must be
    # converted to a mixture before it can branch. The result should match the
    # equivalent one-branch mixture, which reaches the same code with the
    # conversion already done.
    n_quantum = 2
    reg_list = [0]
    pure = _state_initialization_stabilizer(n_quantum)
    assert isinstance(pure.rep_data, Stabilizer)
    one_branch = QuantumState(
        [(1.0, sfc.create_n_ket0_state(n_quantum))], rep_type="s", mixed=True
    )

    noise = nm.DepolarizingNoise(depolarizing_prob)
    noise.apply(pure, n_quantum, reg_list)
    noise.apply(one_branch, n_quantum, reg_list)

    assert pure.mixed
    assert isinstance(pure.rep_data, MixedStabilizer)
    assert np.isclose(pure.rep_data.probability, 1.0)
    got = pure.rep_data.mixture
    expected = one_branch.rep_data.mixture
    assert len(got) == len(expected)
    for (p_got, t_got), (p_expected, t_expected) in zip(got, expected):
        assert np.isclose(p_got, p_expected)
        assert t_got == t_expected
    if depolarizing_prob > 0:
        assert len(got) > 1


def test_general_kraus_error_single_qubit():
    # A bit flip with probability p on a one-qubit |0> state.
    n_quantum = 1
    p = 0.3
    kraus_ops = [np.sqrt(1 - p) * np.eye(2), np.sqrt(p) * dmf.sigmax()]
    state = _state_initialization_dm(n_quantum)
    noise = nm.GeneralKrausError(kraus_ops)

    noise.apply(state, n_quantum, [0])

    assert np.allclose(state.rep_data.data, np.diag([1 - p, p]))


def test_general_kraus_error_applies_independently_to_every_register():
    # The same single-qubit channel on both qubits of |00> factorizes.
    n_quantum = 2
    p = 0.3
    kraus_ops = [np.sqrt(1 - p) * np.eye(2), np.sqrt(p) * dmf.sigmax()]
    state = _state_initialization_dm(n_quantum)
    noise = nm.GeneralKrausError(kraus_ops)

    noise.apply(state, n_quantum, [0, 1])

    single_qubit_result = np.diag([1 - p, p])
    expected = np.kron(single_qubit_result, single_qubit_result)
    assert np.allclose(state.rep_data.data, expected)


def test_general_kraus_error_refuses_operators_wider_than_one_qubit():
    n_quantum = 1
    state = _state_initialization_dm(n_quantum)
    noise = nm.GeneralKrausError([np.eye(4)])

    with pytest.raises(ValueError, match="single-qubit"):
        noise.apply(state, n_quantum, [0])
