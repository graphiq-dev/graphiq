# Copyright (c) 2022-2026 Quantum Bridge Technologies Inc.
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
"""
Resetting a qubit is a Z-basis measurement followed by an X on the target
when the outcome disagrees with the requested state. Nothing else in the
state may move: the measurement has already projected the state, so any
extra write to the measured row's own sign lands on the generator every
other qubit's row still shares.

These tests build a Bell pair -- the measurement of one of its qubits is
probabilistic, which is the branch that must not special-case a sign
rewrite -- and check both intended states against both forced outcomes with
the density matrix of the tableau's signed generators. ``reset_x`` and ``reset_y`` are built
from ``reset_z`` plus a basis change, so they inherit whatever ``reset_z``
guarantees; the tests below hold that guarantee for each in turn.
"""

import numpy as np

import graphiq.backends.state_rep_conversion as rc
from graphiq.backends.stabilizer.functions.clifford import (
    create_n_ket0_state,
    reset_x,
    reset_y,
    reset_z,
)
from graphiq.backends.stabilizer.functions.transformation import (

    cnot_gate,
    hadamard_gate,
)


def tableau_to_density(tableau):
    """The density matrix of a Clifford tableau's stabilizer half, signs included."""
    return rc.stabilizer_to_density(tableau.to_stabilizer())


def _bell_state():
    tableau = create_n_ket0_state(2)
    tableau = hadamard_gate(tableau, 0)
    tableau = cnot_gate(tableau, 0, 1)
    return tableau


def _expect(index):
    expected = np.zeros((4, 4))
    expected[index, index] = 1.0
    return expected


def test_resetting_a_bell_pair_qubit_forced_to_zero_to_ket_0_leaves_ket_00():
    tableau = reset_z(_bell_state(), 0, intended_state=0, measurement_determinism=0)
    assert np.allclose(tableau_to_density(tableau), _expect(0b00))


def test_resetting_a_bell_pair_qubit_forced_to_zero_to_ket_1_leaves_ket_10():
    tableau = reset_z(_bell_state(), 0, intended_state=1, measurement_determinism=0)
    assert np.allclose(tableau_to_density(tableau), _expect(0b10))


def test_resetting_a_bell_pair_qubit_forced_to_one_to_ket_1_leaves_ket_11():
    tableau = reset_z(_bell_state(), 0, intended_state=1, measurement_determinism=1)
    assert np.allclose(tableau_to_density(tableau), _expect(0b11))


def test_resetting_a_bell_pair_qubit_forced_to_one_to_ket_0_leaves_ket_01():
    # The projection has already happened on the forced-1 branch; a reset to
    # |0> here must apply X to the target qubit rather than rewriting the
    # measured row's own sign, which would corrupt the surviving generator
    # that the untouched qubit still shares.
    tableau = reset_z(_bell_state(), 0, intended_state=0, measurement_determinism=1)
    assert np.allclose(tableau_to_density(tableau), _expect(0b01))


def test_resetting_a_bell_pair_qubit_in_x_basis_forced_to_one_to_plus_leaves_plus_1():
    # reset_x is reset_z followed by a Hadamard, so it inherits reset_z's
    # sign-exactness on the probabilistic branch.
    tableau = reset_x(_bell_state(), 0, intended_state=0, measurement_determinism=1)
    plus = np.array([[0.5, 0.5], [0.5, 0.5]])
    one = _expect(1)[:2, :2]
    assert np.allclose(tableau_to_density(tableau), np.kron(plus, one))


def test_resetting_a_bell_pair_qubit_in_y_basis_forced_to_one_to_plus_i_leaves_plus_i_1():
    # reset_y is reset_z followed by H then S, so it inherits the same
    # guarantee. Forcing the outcome to 1 while asking for 0 is what makes this
    # discriminate: the two agree in the cell where the reset has nothing to
    # do, and only disagree where the correcting X is required.
    tableau = reset_y(_bell_state(), 0, intended_state=0, measurement_determinism=1)
    plus_i = np.array([[0.5, -0.5j], [0.5j, 0.5]])
    one = _expect(1)[:2, :2]
    assert np.allclose(tableau_to_density(tableau), np.kron(plus_i, one))
