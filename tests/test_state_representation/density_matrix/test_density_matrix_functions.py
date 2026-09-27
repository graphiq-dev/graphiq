# Copyright (c) 2022-2024 Quantum Bridge Technologies Inc.
# Copyright (c) 2022-2024 Ki3 Photonics Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import pytest

import graphiq.backends.density_matrix.functions as dmf
from graphiq.backends.density_matrix.functions import *


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


def test_swap_two_qubits_exchanges_a_computational_basis_state():
    rho = ket2dm(tensor([state_ketz0(), state_ketz1()]))

    swapped = swap_two_qubits(rho, 0, 1)

    assert np.allclose(swapped, ket2dm(tensor([state_ketz1(), state_ketz0()])))


def test_swap_two_qubits_refuses_equal_control_and_target():
    # Delegates to get_two_qubit_controlled_gate, which refuses to build a controlled gate
    # whose control and target are the same qubit.
    rho = ket2dm(tensor([state_ketz0(), state_ketz1()]))

    with pytest.raises(ValueError):
        swap_two_qubits(rho, 0, 0)


def test_trace_out_qubit_of_a_bell_pair_leaves_the_maximally_mixed_state():
    bell = (
        tensor([state_ketz0(), state_ketz0()]) + tensor([state_ketz1(), state_ketz1()])
    ) / np.sqrt(2)
    rho = ket2dm(bell)

    traced = trace_out_qubit(rho, 1)

    assert np.allclose(traced, np.eye(2) / 2)


def test_trace_out_qubit_of_a_product_state_leaves_the_untraced_factor():
    rho = ket2dm(tensor([state_ketz0(), state_ketz1()]))

    traced = trace_out_qubit(rho, 1)

    assert np.allclose(traced, ket2dm(state_ketz0()))


def test_partial_trace_of_a_bell_pair_is_maximally_mixed():
    # Tracing one half of |00> + |11> must remove the coherence between |0> and
    # |1>. Summing the traced qubit's row and column indices separately instead of
    # along its diagonal leaves |+><+| here, off-diagonals of 0.5 instead of 0.
    bell = np.array([1, 0, 0, 1]) / np.sqrt(2)
    rho = np.outer(bell, bell.conj())

    for keep in ([0], [1]):
        assert np.allclose(partial_trace(rho, keep, [2, 2]), np.eye(2) / 2)


@pytest.mark.parametrize("keep", [[0], [1], [2], [0, 1], [0, 2], [1, 2]])
def test_partial_trace_matches_an_explicit_sum_over_traced_basis_states(keep):
    # Reference: sum over the traced qubits' basis states of <k| rho |k>, built
    # with explicit projectors, on a random entangled three-qubit state.
    rng = np.random.default_rng(7)
    psi = rng.normal(size=8) + 1j * rng.normal(size=8)
    psi /= np.linalg.norm(psi)
    rho = np.outer(psi, psi.conj())

    traced = [q for q in range(3) if q not in keep]
    expected = np.zeros((2 ** len(keep), 2 ** len(keep)), dtype=complex)
    for bits in np.ndindex(*(2,) * len(traced)):
        # a (8 x 2**len(keep)) isometry picking out the traced qubits' values
        iso = np.zeros((8, 2 ** len(keep)))
        for kept_bits in np.ndindex(*(2,) * len(keep)):
            full = [0, 0, 0]
            for q, b in zip(traced, bits):
                full[q] = b
            for q, b in zip(keep, kept_bits):
                full[q] = b
            row = full[0] * 4 + full[1] * 2 + full[2]
            col = int("".join(map(str, kept_bits)), 2) if keep else 0
            iso[row, col] = 1
        expected += iso.T @ rho @ iso

    assert np.allclose(partial_trace(rho, keep, [2, 2, 2]), expected)


# two non-commuting mixed qubit states, both complex
_RHO = np.array([[0.5, -0.2j], [0.2j, 0.5]])
_SIGMA = np.array([[0.6, 0.1 - 0.1j], [0.1 + 0.1j, 0.4]])
# for qubits, F = tr(rho sigma) + 2 sqrt(det rho det sigma)
_F_RHO_SIGMA = 0.54 + 2 * np.sqrt(0.21 * 0.22)


def test_fidelity_of_two_different_real_mixed_states():
    p, q = np.array([0.7, 0.3]), np.array([0.5, 0.5])
    # commuting states: F = (sum_i sqrt(p_i q_i))^2
    expected = np.sum(np.sqrt(p * q)) ** 2

    assert fidelity(np.diag(p), np.diag(q)) == pytest.approx(expected)


def test_fidelity_of_a_complex_mixed_state_with_itself_is_one():
    assert fidelity(_RHO, _RHO.copy()) == pytest.approx(1.0)


def test_fidelity_of_two_complex_mixed_states():
    assert fidelity(_RHO, _SIGMA) == pytest.approx(_F_RHO_SIGMA)


def test_sqrtm_psd_of_a_complex_matrix_squares_back_to_it():
    root = sqrtm_psd(_RHO)

    assert np.allclose(root @ root, _RHO)


def test_sqrtm_psd_of_a_real_valued_matrix_is_unchanged():
    # complex dtype with zero imaginary parts, as the compiler's states carry
    matrix = np.array([[0.6, 0.2], [0.2, 0.4]], dtype=complex)
    eig_vals, eig_vecs = eigh(matrix)

    expected = (eig_vecs * np.sqrt(np.maximum(eig_vals, 0))) @ eig_vecs.T

    assert np.array_equal(sqrtm_psd(matrix), expected)


def test_fidelity_still_raises_above_one(monkeypatch):
    # fidelity looks sqrtm_psd up in its own module, so patch it there
    monkeypatch.setattr(dmf, "sqrtm_psd", lambda m: 1.1 * np.eye(2))

    with pytest.raises(Warning):
        fidelity(np.diag([0.7, 0.3]), np.diag([0.5, 0.5]))


def _elementwise_partial_transpose(rho, dim1, dim2, subsys):
    # the definition: with row index (i, j) and column index (k, l), transposing
    # subsystem 0 exchanges i and k, and transposing subsystem 1 exchanges j and l
    out = np.zeros_like(rho)
    for i in range(dim1):
        for j in range(dim2):
            for k in range(dim1):
                for l in range(dim2):
                    if subsys == 0:
                        out[k * dim2 + j, i * dim2 + l] = rho[
                            i * dim2 + j, k * dim2 + l
                        ]
                    else:
                        out[i * dim2 + l, k * dim2 + j] = rho[
                            i * dim2 + j, k * dim2 + l
                        ]
    return out


@pytest.mark.parametrize("subsys", [0, 1])
@pytest.mark.parametrize("dim1, dim2", [(2, 2), (3, 3), (2, 3), (3, 2), (2, 4), (4, 2)])
def test_bipartite_partial_transpose_matches_the_elementwise_definition(
    dim1, dim2, subsys
):
    rng = np.random.default_rng(0)
    a = rng.normal(size=(dim1 * dim2, dim1 * dim2)) + 1j * rng.normal(
        size=(dim1 * dim2, dim1 * dim2)
    )
    rho = a @ a.conj().T
    rho = rho / np.trace(rho)

    assert np.allclose(
        bipartite_partial_transpose(rho, dim1, dim2, subsys),
        _elementwise_partial_transpose(rho, dim1, dim2, subsys),
    )


def test_bipartite_partial_transpose_of_a_product_transposes_one_factor():
    rng = np.random.default_rng(1)
    a = rng.normal(size=(2, 2))
    b = rng.normal(size=(3, 3))

    assert np.allclose(
        bipartite_partial_transpose(np.kron(a, b), 2, 3, 0), np.kron(a.T, b)
    )
    assert np.allclose(
        bipartite_partial_transpose(np.kron(a, b), 2, 3, 1), np.kron(a, b.T)
    )


def test_negativity_with_unequal_dimensions():
    # a Bell pair on qubits 0 and 1, with qubit 2 in |0>, split as qubit 0 | qubits 1, 2
    bell = np.array([1, 0, 0, 1]) / np.sqrt(2)
    state = np.kron(bell, np.array([1, 0]))
    assert np.isclose(negativity(np.outer(state, state.conj()), 2, 4), 0.5)

    # a product state across the same 2 | 4 split has no negativity
    plus = np.array([1, 1]) / np.sqrt(2)
    phi = np.array([1, 0, 1j, 0]) / np.sqrt(2)
    product = np.kron(np.outer(plus, plus.conj()), np.outer(phi, phi.conj()))
    assert np.isclose(negativity(product, 2, 4), 0, atol=1e-12)
