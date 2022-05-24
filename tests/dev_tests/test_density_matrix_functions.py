import pytest

from src.backends.density_matrix.functions import *

n_qubits = 8
q_register = 0
state = ketx0_state()

state = reduce(np.kron, n_qubits * [state @ np.conjugate(state.T)])
m0, m1 = projectors_zbasis(n_qubits, q_register)
outcome0 = np.trace(state @ m0)
outcome1 = np.trace(state @ m1)

assert np.isclose(outcome0, 0.5)
assert np.isclose(outcome1, 0.5)
