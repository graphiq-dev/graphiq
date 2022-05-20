import pytest

from src.backends.density_matrix.functions import *


@pytest.mark.parametrize("n_qubits", "q_register", "input_state",
                         [(1, 0, 'z1', (0, 1))])
def test_measurements(n_qubits, q_register, input_state):
    if input_state == "z0":
        state = ketz0_state()
    else:
        raise ValueError

    state = reduce(np.kron, n_qubits * [state @ np.conjugate(state.T)])
    m0, m1 = projectors_zbasis(n_qubits, q_register)
    # print(m0)
    # print(m1)
    print(state)
    res = np.trace(state @ m0)
    # print(res)

    res = np.trace(state @ m1)
    # print(res)

    return


