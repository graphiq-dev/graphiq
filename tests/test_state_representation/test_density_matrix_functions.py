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


def test_controlled_gate():
    assert np.array_equal(
        get_controlled_gate(4, 1, 2, sigmaz()),
        get_controlled_gate_efficient(4, 1, 2, sigmaz()),
    )
    assert np.array_equal(
        get_controlled_gate(4, 1, 2, sigmax()),
        get_controlled_gate_efficient(4, 1, 2, sigmax()),
    )
    assert np.array_equal(
        get_controlled_gate(5, 2, 4, sigmaz()),
        get_controlled_gate_efficient(5, 2, 4, sigmaz()),
    )
    assert np.array_equal(
        get_controlled_gate(5, 4, 1, sigmay()),
        get_controlled_gate_efficient(5, 4, 1, sigmay()),
    )
