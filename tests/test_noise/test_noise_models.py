import numpy as np
import numpy.testing as nptest
import pytest

from src import ops

import src.backends.density_matrix.functions as dmf
import src.noise.noise_models as nm
from src.state import QuantumState
from src.backends.state_base import StateRepresentationBase
from src.backends.density_matrix.state import DensityMatrix
from src.backends.stabilizer.state import Stabilizer
from src.backends.graph.state import Graph


def _state_initialization(n_quantum, state=dmf.state_ketz0()):
    return QuantumState(
        n_quantum,
        dmf.create_n_product_state(n_quantum, state),
        representation="density matrix",
    )


def test_noise_base():
    test_noise = nm.NoiseBase()
    assert test_noise.noise_parameters == {}

    with pytest.raises(NotImplementedError):
        test_noise.get_backend_dependent_noise()


def test_addition_noise_base():
    test_noise = nm.AdditionNoiseBase()
    assert test_noise.noise_parameters == {'After gate': True}

    with pytest.raises(NotImplementedError):
        test_noise.get_backend_dependent_noise()


def test_replacement_noise_base():
    test_noise = nm.ReplacementNoiseBase()
    assert test_noise.noise_parameters == {}

    with pytest.raises(NotImplementedError):
        test_noise.get_backend_dependent_noise()


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


# No error
@pytest.mark.parametrize(
    "state_rep, expected_output",
    [
        (DensityMatrix(data=1), np.eye(dmf.create_n_product_state(1, dmf.state_ketz0()).shape[0])),
        (Stabilizer(data=1), []),
        (Graph(data=1, root_node_id=1), None)
    ],
)
def test_no_noise_get_backend(state_rep, expected_output):
    test_noise = nm.NoNoise()
    output = test_noise.get_backend_dependent_noise(state_rep, n_quantum=1)
    if isinstance(output, np.ndarray):
        nptest.assert_equal(output, expected_output)
    else:
        assert output == expected_output


# With error
def test_no_noise_get_backend_error():
    test_noise = nm.NoNoise()
    state_rep = StateRepresentationBase(data=1)

    with pytest.raises(TypeError, match="Backend type is not supported."):
        test_noise.get_backend_dependent_noise(state_rep=state_rep, n_quantum=1)


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
        noise.get_backend_dependent_noise(state1.dm, n_quantum, [qubit_position]),
        hadamard_gate,
    )
    noise.apply(state1, n_quantum, [qubit_position])

    state2.dm.apply_unitary(hadamard_gate)
    assert np.allclose(state1.dm.data, state2.dm.data)


@pytest.mark.parametrize(
    "state_rep, expected_output",
    [
        (Stabilizer(5), None),
        (Graph(5, root_node_id=1), None),
    ]
)
def test_one_qubit_replacement_get_backends(state_rep, expected_output):
    noise = nm.OneQubitGateReplacement(
        dmf.parameterized_one_qubit_unitary(np.pi / 2, 0, np.pi)
    )

    result = noise.get_backend_dependent_noise(state_rep=state_rep, n_quantum=5, reg_list=[2])
    assert result == expected_output


def test_one_qubit_replacement_error():
    noise = nm.OneQubitGateReplacement(
        dmf.parameterized_one_qubit_unitary(np.pi / 2, 0, np.pi)
    )

    state_rep = StateRepresentationBase(data=5)
    with pytest.raises(TypeError, match="Backend type is not supported."):
        noise.get_backend_dependent_noise(state_rep=state_rep, n_quantum=5, reg_list=[2])


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
            state1.dm, n_quantum, control_qubit, target_qubit
        ),
        cnot_gate,
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
    assert np.allclose(
        noise.get_backend_dependent_noise(state1.dm, n_quantum, [qubit_position]),
        x_gate,
    )
    noise.apply(state1, n_quantum, [qubit_position])

    state2.dm.apply_unitary(x_gate)
    assert np.allclose(state1.dm.data, state2.dm.data)
