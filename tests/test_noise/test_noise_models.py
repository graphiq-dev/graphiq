import numpy as np
from src import ops

import src.backends.density_matrix.functions as dmf
import src.noise.noise_models as nm
from src.backends.density_matrix.state import DensityMatrix


def _state_initialization(n_quantum, state=dmf.state_ketz0()):
    return DensityMatrix(data=dmf.create_n_product_state(n_quantum, state))


def test_depolarizing_noise():
    n_quantum = 3
    qubit_position = 1
    qubit_state = dmf.state_ketx0()
    state1 = _state_initialization(n_quantum, qubit_state)
    state2 = _state_initialization(n_quantum, qubit_state)
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
    state2.apply_channel(kraus_ops)

    assert np.allclose(state1.data, state2.data)


def test_no_noise():
    n_quantum = 5
    qubit_state = dmf.state_kety0()
    state1 = _state_initialization(n_quantum, qubit_state)
    state2 = _state_initialization(n_quantum, qubit_state)
    noise = nm.NoNoise()
    noise.apply(state1)

    assert np.allclose(state1.data, state2.data)


def test_one_qubit_replacement():
    n_quantum = 5
    qubit_position = 2
    state1 = _state_initialization(n_quantum, dmf.state_ketx0())
    state2 = _state_initialization(n_quantum, dmf.state_ketx0())
    noise = nm.OneQubitGateReplacement(
        dmf.parameterized_one_qubit_unitary(np.pi / 2, 0, np.pi)
    )

    hadamard_gate = dmf.get_one_qubit_gate(n_quantum, qubit_position, dmf.hadamard())
    assert np.allclose(
        noise.get_backend_dependent_noise(state1, n_quantum, [qubit_position]),
        hadamard_gate,
    )
    noise.apply(state1, n_quantum, [qubit_position])

    state2.apply_unitary(hadamard_gate)
    assert np.allclose(state1.data, state2.data)


def test_two_qubit_replacement():
    n_quantum = 5
    control_qubit = 2
    target_qubit = 0
    state1 = _state_initialization(n_quantum, dmf.state_ketx1())
    state2 = _state_initialization(n_quantum, dmf.state_ketx1())
    noise = nm.TwoQubitControlledGateReplacement(
        dmf.parameterized_one_qubit_unitary(np.pi, 0, np.pi), phase_factor=0
    )

    cnot_gate = dmf.get_two_qubit_controlled_gate(
        n_quantum, control_qubit, target_qubit, dmf.sigmax()
    )

    assert np.allclose(
        noise.get_backend_dependent_noise(
            state1, n_quantum, control_qubit, target_qubit
        ),
        cnot_gate,
    )
    noise.apply(state1, n_quantum, control_qubit, target_qubit)

    state2.apply_unitary(cnot_gate)
    assert np.allclose(state1.data, state2.data)


def test_pauli_error():
    n_quantum = 5
    qubit_position = 2
    state1 = _state_initialization(n_quantum, dmf.state_ketx0())
    state2 = _state_initialization(n_quantum, dmf.state_ketx0())
    noise = nm.PauliError("X")

    x_gate = dmf.get_one_qubit_gate(n_quantum, qubit_position, dmf.sigmax())
    assert np.allclose(
        noise.get_backend_dependent_noise(state1, n_quantum, [qubit_position]),
        x_gate,
    )
    noise.apply(state1, n_quantum, [qubit_position])

    state2.apply_unitary(x_gate)
    assert np.allclose(state1.data, state2.data)
