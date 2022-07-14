import pytest
import numpy as np
from src.backends.density_matrix.functions import *


def test_measurement_prob():
    n_qubits = 8
    q_register = 0
    st = ketx0_state()

    st = reduce(np.kron, n_qubits * [st @ np.conjugate(st.T)])
    m0, m1 = projectors_zbasis(n_qubits, q_register)
    outcome0 = np.trace(st @ m0)
    outcome1 = np.trace(st @ m1)

    assert np.isclose(outcome0, 0.5)
    assert np.isclose(outcome1, 0.5)


def test_single_qubit_unitary():
    assert np.allclose(one_qubit_unitary(2, 1, 0, 0, 0), np.eye(4))
    assert np.allclose(one_qubit_unitary(1, 1, 0, 0, np.pi / 2), phase())
    assert np.allclose(one_qubit_unitary(1, 1, np.pi / 2, 0, np.pi), hadamard())
    assert np.allclose(
        one_qubit_unitary(2, 1, 0, 0, np.pi / 2),
        get_single_qubit_gate(2, 1, phase()),
    )
    assert np.allclose(
        one_qubit_unitary(2, 1, np.pi / 2, 0, np.pi),
        get_single_qubit_gate(2, 1, hadamard()),
    )


def test_multi_qubit_gate():

    qubit_positions = [1, 3]
    target_gates = [sigmax(), sigmaz()]
    assert np.allclose(
        get_multi_qubit_gate(4, qubit_positions, target_gates),
        get_single_qubit_gate(4, 1, sigmax()) @ get_single_qubit_gate(4, 3, sigmaz()),
    )

    assert np.allclose(
        get_multi_qubit_gate(5, qubit_positions, target_gates),
        get_single_qubit_gate(5, 1, sigmax()) @ get_single_qubit_gate(5, 3, sigmaz()),
    )


def test_multi_qubit_gate2():

    assert np.allclose(
        get_multi_qubit_gate(5, [4, 3], [sigmax(), hadamard()]),
        get_single_qubit_gate(5, 4, sigmax()) @ get_single_qubit_gate(5, 3, hadamard()),
    )


def test_multi_qubit_gate3():

    assert np.allclose(
        get_multi_qubit_gate(1, [0], [sigmax()]),
        get_single_qubit_gate(1, 0, sigmax()),
    )
