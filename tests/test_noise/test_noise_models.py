import numpy as np
from src import ops

import src.backends.density_matrix.functions as dmf
import src.backends.stabilizer.functions.clifford as sfc
import src.noise.noise_models as nm
from src.state import QuantumState


def _state_initialization(n_quantum, state=dmf.state_ketz0()):
    return QuantumState(
        n_quantum,
        dmf.create_n_product_state(n_quantum, state),
        representation="density matrix",
    )


def _state_initialization_stabilizer(n_quantum):
    return QuantumState(
        n_quantum,
        sfc.create_n_ket0_state(n_quantum),
        representation="stabilizer",
    )


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
    state2.dm.apply_channel(kraus_ops)

    assert np.allclose(state1.dm.data, state2.dm.data)


def test_no_noise():
    n_quantum = 5
    qubit_state = dmf.state_kety0()
    state1 = _state_initialization(n_quantum, qubit_state)
    state2 = _state_initialization(n_quantum, qubit_state)
    noise = nm.NoNoise()
    noise.apply(state1)

    assert np.allclose(state1.dm.data, state2.dm.data)


def test_one_qubit_replacement():
    n_quantum = 5
    qubit_position = 2
    state1 = _state_initialization(n_quantum, dmf.state_ketx0())
    state2 = _state_initialization(n_quantum, dmf.state_ketx0())
    noise = nm.OneQubitGateReplacement(
        dmf.parameterized_one_qubit_unitary(np.pi / 2, 0, np.pi)
    )

    hadamard_gate = dmf.get_one_qubit_gate(n_quantum, qubit_position, dmf.hadamard())

    noise.apply(state1, n_quantum, [qubit_position])

    state2.dm.apply_unitary(hadamard_gate)
    assert np.allclose(state1.dm.data, state2.dm.data)


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

    noise.apply(state1, n_quantum, control_qubit, target_qubit)

    state2.dm.apply_unitary(cnot_gate)
    assert np.allclose(state1.dm.data, state2.dm.data)


def test_pauli_error():
    n_quantum = 5
    qubit_position = 2
    state1 = _state_initialization(n_quantum, dmf.state_ketx0())
    state2 = _state_initialization(n_quantum, dmf.state_ketx0())
    noise = nm.PauliError("X")

    x_gate = dmf.get_one_qubit_gate(n_quantum, qubit_position, dmf.sigmax())
    noise.apply(state1, n_quantum, [qubit_position])

    state2.dm.apply_unitary(x_gate)
    assert np.allclose(state1.dm.data, state2.dm.data)


def test_photon_loss():
    n_quantum = 2
    state1 = _state_initialization_stabilizer(n_quantum)
    noise = nm.PhotonLoss(0.1)
    noise.apply(state1, n_quantum, 1)
    print(state1.stabilizer)
