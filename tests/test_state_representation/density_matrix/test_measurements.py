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
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pytest

import graphiq.backends.density_matrix.functions as dmf
import graphiq.backends.state_rep_conversion as rc
import graphiq.circuit.ops as ops
from graphiq.backends.density_matrix.compiler import DensityMatrixCompiler
from graphiq.backends.density_matrix.state import (
    DensityMatrix,
    _forced_measurement_tolerance,
)
from graphiq.circuit.circuit_dag import CircuitDAG


def test_y_measurement():
    graph1 = nx.Graph([(1, 2), (2, 3), (3, 4), (1, 3)])
    rho = DensityMatrix(rc.graph_to_density(graph1))
    kety0 = dmf.state_kety0()
    kety1 = dmf.state_kety1()
    projector0 = dmf.get_one_qubit_gate(4, 2, kety0 @ np.conjugate(kety0.T))
    projector1 = dmf.get_one_qubit_gate(4, 2, kety1 @ np.conjugate(kety1.T))
    projectors = [projector0, projector1]
    rho.apply_measurement(projectors, measurement_determinism=0)
    graph_adj = rc.density_to_graph(rho.data, validate=False)
    nx.draw(nx.from_numpy_array(graph_adj), with_labels=True)
    plt.show()


def test_x_measurement():
    graph1 = nx.Graph([(1, 2), (2, 3), (3, 4), (1, 3), (1, 4), (2, 4), (1, 5), (2, 5)])
    # nx.draw(graph1, with_labels=True)
    rho = DensityMatrix(rc.graph_to_density(graph1))
    ketx0 = dmf.state_ketx0()
    ketx1 = dmf.state_ketx1()
    graph2 = nx.Graph([(1, 2), (1, 4), (2, 4), (1, 5), (2, 5)])
    graph2.add_node(3)
    nx.draw(graph2, with_labels=True)
    rho2 = DensityMatrix(rc.graph_to_density(graph2))
    projector0 = dmf.get_one_qubit_gate(5, 2, ketx0 @ np.conjugate(ketx0.T))
    projector1 = dmf.get_one_qubit_gate(5, 2, ketx1 @ np.conjugate(ketx1.T))
    projectors = [projector0, projector1]
    rho.apply_measurement(projectors, measurement_determinism=1)
    print(f"rho is {rho.data}")
    print(f"rho2 is {rho2.data}")

    graph_adj = rc.density_to_graph(rho.data, validate=False)
    # nx.draw(nx.from_numpy_array(graph_adj),with_labels=True)
    plt.show()


_PLUS = np.full((2, 2), 0.5)
_Z_PROJECTORS = [np.diag([1.0, 0.0]), np.diag([0.0, 1.0])]


@pytest.mark.parametrize("determinism", [0, 1, "probabilistic"])
def test_measurement_keeps_the_trace_of_a_subnormalised_state(determinism):
    # photon loss leaves a trace below 1, which the measurement must keep
    np.random.seed(0)
    rho = DensityMatrix(0.9 * _PLUS, normalized=False)

    outcome = rho.apply_measurement(_Z_PROJECTORS, measurement_determinism=determinism)

    assert np.allclose(rho.data, 0.9 * _Z_PROJECTORS[outcome])


@pytest.mark.parametrize("determinism", [0, 1])
def test_measurement_keeps_a_trace_just_below_one(determinism):
    # a loss of 1e-6 is far above rounding drift, so it is kept too
    rho = DensityMatrix(0.999999 * _PLUS, normalized=False)

    rho.apply_measurement(_Z_PROJECTORS, measurement_determinism=determinism)

    assert np.real(np.trace(rho.data)) == pytest.approx(0.999999, rel=1e-12)


def _path3_with_trace_one_up_to_rounding():
    # four gate and dephasing steps leave the trace at 1 - 1.4e-15
    h = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    unitaries = [np.kron(np.eye(2), np.kron(h, h)), np.kron(h, np.eye(4))]
    dephasing = [np.sqrt(0.9) * np.eye(8), np.sqrt(0.1) * np.diag([1, -1] * 4)]
    rho = DensityMatrix(rc.graph_to_density(nx.path_graph(3)))
    for step in range(4):
        rho.apply_unitary(unitaries[step % 2])
        rho.apply_channel(dephasing)
    return rho


@pytest.mark.parametrize("determinism", [0, 1])
def test_measurement_of_a_state_with_trace_one_up_to_rounding_is_unchanged(
    determinism,
):
    rho = _path3_with_trace_one_up_to_rounding()
    before = rho.data.copy()
    trace = np.real(np.trace(before))
    assert np.trace(before) != 1.0
    projectors = [
        np.kron(np.diag([1.0, 0.0]), np.eye(4)),
        np.kron(np.diag([0.0, 1.0]), np.eye(4)),
    ]

    outcome = rho.apply_measurement(projectors, measurement_determinism=determinism)

    m = projectors[outcome]
    expected = (m @ before @ np.transpose(np.conjugate(m))) / np.real(
        np.trace(before @ m)
    )
    # scaling by this trace would change the matrix, so the check can tell
    assert not np.array_equal(trace * expected, expected)
    assert np.array_equal(rho.data, expected)


# A forced outcome is taken only if its probability is above a resolution floor,
# tol = dim * eps * sum(|probs|); below it the probability is rounding error. These
# build states with normalized=False, since the constructor renormalises the very
# roundoff under test.


def _forced(rho, determinism, n_qubits=1):
    state = DensityMatrix(np.asarray(rho, dtype=complex), normalized=False)
    outcome = state.apply_measurement(
        dmf.projectors_zbasis(n_qubits, 0), measurement_determinism=determinism
    )
    return outcome, state.data


def _skewed_block(trace, coherence=3e-16):
    # a 2x2 block with the given trace and a fixed off-diagonal roundoff: indefinite for
    # a trace well below the coherence, so dividing by that trace would amplify a
    # negative eigenvalue to order one; positive semidefinite well above it
    a = b = trace / 2
    return np.array([[a, coherence], [coherence, b]])


def _forced_boundary_state(forced_index, forced_trace, other_trace):
    # 2 qubits, block diagonal: the forced outcome's block is _skewed_block(forced_trace)
    # and the other outcome's block is diag([other_trace, 0])
    forced_block = _skewed_block(forced_trace)
    other_block = np.diag([other_trace, 0.0])
    rho = np.zeros((4, 4), dtype=complex)
    if forced_index == 0:
        rho[:2, :2], rho[2:, 2:] = forced_block, other_block
    else:
        rho[:2, :2], rho[2:, 2:] = other_block, forced_block
    return rho


def test_a_rounding_error_probability_is_not_forced():
    # Hadamard twice is the identity, so the state is |0>, but p(1) comes out as rounding
    # error; the precondition makes the test fail loudly, rather than pass without
    # testing anything, on a platform whose roundoff differs
    circuit = CircuitDAG(n_emitter=1, n_classical=0)
    circuit.add(ops.Hadamard(register=0, reg_type="e"))
    circuit.add(ops.Hadamard(register=0, reg_type="e"))
    state = DensityMatrixCompiler().compile(circuit).rep_data
    projectors = dmf.projectors_zbasis(1, 0)
    assert 0 < np.real(np.trace(state.data @ projectors[1])) < 1e-20

    outcome = state.apply_measurement(projectors, measurement_determinism=1)

    assert outcome == 0
    assert np.allclose(state.data, np.diag([1.0, 0.0]))


@pytest.mark.parametrize("over_one", [False, True])
@pytest.mark.parametrize("k", [0.5, 2, 10])
@pytest.mark.parametrize("determinism", [0, 1])
def test_forced_measurement_boundary_is_where_the_code_computes_it(
    determinism, k, over_one
):
    # The forced outcome's block trace is k * tol, with tol computed by the code's own
    # function, so a small shift in roundoff does not flip the test; only a change to the
    # tolerance does. Below tol the forced outcome is not taken (the block is indefinite
    # there); above it, it is, and the result is still a valid density matrix. over_one
    # puts the trace just above 1 (and, for k <= 2, the other outcome's probability),
    # which must not matter. tol is taken for a total of 1, which the state's total
    # matches to within 4e-15.
    tol = _forced_measurement_tolerance(4, np.array([1.0]))
    forced_trace = k * tol
    other_trace = 1.0 - forced_trace + (3.55e-15 if over_one else 0.0)
    rho = _forced_boundary_state(determinism, forced_trace, other_trace)

    outcome, data = _forced(rho, determinism, n_qubits=2)

    assert outcome == (determinism if k > 1 else 1 - determinism)
    assert dmf.is_density_matrix(data)


def test_forcing_zero_on_a_state_with_trace_above_one():
    # an unnormalised state: p(1) = 1 does not make outcome 0 impossible
    outcome, _ = _forced(np.diag([1.0, 1.0]), 0)

    assert outcome == 0


@pytest.mark.parametrize("n_qubits", [1, 2, 3, 4])
def test_a_state_is_psd_accepts_is_still_measurable(n_qubits):
    # every eigenvalue is within is_psd's -1e-15, but the outcome-1 block sums to
    # -0.9e-15 * 2^(n-1); a negative probability counts as 0
    dim = 2**n_qubits
    diagonal = np.zeros(dim)
    diagonal[dim // 2 :] = -0.9e-15
    diagonal[0] = 1 + 0.9e-15 * (dim // 2)
    rho = np.diag(diagonal)
    assert dmf.is_density_matrix(rho)

    for determinism in (0, 1, "probabilistic"):
        outcome, _ = _forced(rho, determinism, n_qubits=n_qubits)
        assert outcome == 0


def test_a_branch_far_above_tolerance_is_honoured():
    # theta = pi - 1e-6 gives p(0) = cos^2(theta / 2) ~= 2.5e-13, far above tol (~4.4e-16
    # at one qubit): a real branch, which a coarse threshold such as 1e-9 would discard
    circuit = CircuitDAG(n_emitter=1, n_classical=0)
    circuit.add(
        ops.ParameterizedOneQubitRotation(
            register=0, reg_type="e", params=(np.pi - 1e-6, 0.0, 0.0)
        )
    )
    state = DensityMatrixCompiler().compile(circuit).rep_data

    outcome = state.apply_measurement(
        dmf.projectors_zbasis(1, 0), measurement_determinism=0
    )

    assert outcome == 0
    assert dmf.is_density_matrix(state.data)


def test_an_int64_state_does_not_crash():
    # normalized=False keeps an integer matrix as it is
    state = DensityMatrix(np.array([[1, 0], [0, 0]]), normalized=False)

    outcome = state.apply_measurement(
        dmf.projectors_zbasis(1, 0), measurement_determinism=0
    )

    assert outcome == 0


def test_an_impossible_forced_outcome_returns_the_other_one():
    assert _forced(np.diag([1.0, 0.0]), 1)[0] == 0
    assert _forced(np.diag([0.0, 1.0]), 0)[0] == 1


def test_an_ordinary_forced_measurement_is_unaffected():
    assert _forced(np.diag([0.5, 0.5]), 0)[0] == 0
    assert _forced(np.diag([0.5, 0.5]), 1)[0] == 1


def test_a_redirected_forced_outcome_keeps_a_subnormalised_trace():
    outcome, data = _forced(np.diag([0.5, 0.0]), 1)

    assert outcome == 0
    assert np.real(np.trace(data)) == pytest.approx(0.5)
