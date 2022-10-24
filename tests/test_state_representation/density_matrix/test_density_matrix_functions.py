import pytest
import numpy as np
from src.backends.density_matrix.functions import *


def test_measurement_prob():
    n_qubits = 8
    q_register = 0
    st = state_ketx0()

    st = reduce(np.kron, n_qubits * [st @ np.conjugate(st.T)])
    m0, m1 = projectors_zbasis(n_qubits, q_register)
    outcome0 = np.trace(st @ m0)
    outcome1 = np.trace(st @ m1)

    assert np.isclose(outcome0, 0.5)
    assert np.isclose(outcome1, 0.5)


def test_single_qubit_unitary():
    assert np.allclose(full_one_qubit_unitary(2, 1, 0, 0, 0), np.eye(4))
    assert np.allclose(full_one_qubit_unitary(1, 1, 0, 0, np.pi / 2), phase())
    assert np.allclose(full_one_qubit_unitary(1, 1, np.pi / 2, 0, np.pi), hadamard())
    assert np.allclose(
        full_one_qubit_unitary(2, 1, 0, 0, np.pi / 2),
        get_one_qubit_gate(2, 1, phase()),
    )
    assert np.allclose(
        full_one_qubit_unitary(2, 1, np.pi / 2, 0, np.pi),
        get_one_qubit_gate(2, 1, hadamard()),
    )


def test_multi_qubit_gate():
    qubit_positions = [1, 3]
    target_gates = [sigmax(), sigmaz()]
    assert np.allclose(
        get_multi_qubit_gate(4, qubit_positions, target_gates),
        get_one_qubit_gate(4, 1, sigmax()) @ get_one_qubit_gate(4, 3, sigmaz()),
    )

    assert np.allclose(
        get_multi_qubit_gate(5, qubit_positions, target_gates),
        get_one_qubit_gate(5, 1, sigmax()) @ get_one_qubit_gate(5, 3, sigmaz()),
    )


def test_multi_qubit_gate2():
    assert np.allclose(
        get_multi_qubit_gate(5, [4, 3], [sigmax(), hadamard()]),
        get_one_qubit_gate(5, 4, sigmax()) @ get_one_qubit_gate(5, 3, hadamard()),
    )


def test_multi_qubit_gate3():
    assert np.allclose(
        get_multi_qubit_gate(1, [0], [sigmax()]),
        get_one_qubit_gate(1, 0, sigmax()),
    )


def test_multi_qubit_gate4():
    assert np.allclose(
        get_multi_qubit_gate(2, [1, 0], [sigmax(), sigmaz()]),
        get_one_qubit_gate(2, 1, sigmax()) @ get_one_qubit_gate(2, 0, sigmaz()),
    )


def test_multi_qubit_gate5():
    assert np.allclose(
        get_multi_qubit_gate(4, [3, 1], [sigmax(), sigmaz()]),
        get_one_qubit_gate(4, 3, sigmax()) @ get_one_qubit_gate(4, 1, sigmaz()),
    )


def test_fidelity():
    rho = 1 / 2 * (ket2dm(state_ketx0()) + ket2dm(state_ketx1()))
    sigma = np.eye(2) / 2
    assert np.allclose(fidelity(rho, sigma), 1.0)
    assert np.allclose(fidelity(rho, ket2dm(state_ketx0())), 0.5)
    assert np.allclose(fidelity(ket2dm(state_ketz0()), ket2dm(state_ketx0())), 0.5)
    assert np.allclose(fidelity(ket2dm(state_ketx0()), ket2dm(state_ketx1())), 0)


@pytest.mark.parametrize("n_qubits", [2, 4, 6])
def test_fidelity_multiqubit(n_qubits):
    rho = create_n_product_state(n_qubits, state_ketx0())
    sigma = (
        create_n_product_state(n_qubits, state_kety0()) / 2
        + create_n_product_state(n_qubits, state_kety1()) / 2
    )
    # works only for even number of qubits
    assert np.allclose(fidelity(rho, sigma), 0.5**n_qubits)
