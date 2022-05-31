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
