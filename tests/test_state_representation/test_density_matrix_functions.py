import pytest

from src.backends.density_matrix.functions import *


def test_trace_1():
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
    assert np.allclose(single_qubit_unitary(2, 1, 0, 0, 0), np.eye(4))
    assert np.allclose(single_qubit_unitary(1, 1, 0, 0, np.pi / 2), phase())
    assert np.allclose(single_qubit_unitary(1, 1, np.pi / 2, 0, np.pi), hadamard())
    assert np.allclose(
        single_qubit_unitary(2, 1, 0, 0, np.pi / 2),
        get_single_qubit_gate(2, 1, phase()),
    )
    assert np.allclose(
        single_qubit_unitary(2, 1, np.pi / 2, 0, np.pi),
        get_single_qubit_gate(2, 1, hadamard()),
    )
