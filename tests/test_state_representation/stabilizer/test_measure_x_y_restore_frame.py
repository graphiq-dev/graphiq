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
``measure_x`` and ``measure_y`` rotate a qubit into the Z frame, measure it
there, and must rotate it back before returning: the transformation gates
mutate the tableau in place, so a caller holding onto the tableau otherwise
keeps a state that is still expressed in the rotated frame -- which reads
back as a different state in the qubit's own basis.

These tests build small states, some entangled, and check the caller's
tableau after the call against the density matrix of the tableau's signed generators. Both
outcomes are forced with ``measurement_determinism`` so nothing here depends
on the unseeded global RNG.
"""

import numpy as np

import graphiq.backends.state_rep_conversion as rc
from graphiq.backends.stabilizer.functions.clifford import (
    create_n_ket0_state,
    measure_x,
    measure_y,
)
from graphiq.backends.stabilizer.functions.transformation import (

    cnot_gate,
    hadamard_gate,
)


def tableau_to_density(tableau):
    """The density matrix of a Clifford tableau's stabilizer half, signs included."""
    return rc.stabilizer_to_density(tableau.to_stabilizer())


_PLUS = np.array([[0.5, 0.5], [0.5, 0.5]])
_MINUS = np.array([[0.5, -0.5], [-0.5, 0.5]])
_PLUS_Y = np.array([[0.5, -0.5j], [0.5j, 0.5]])
_MINUS_Y = np.array([[0.5, 0.5j], [-0.5j, 0.5]])


def test_measuring_ket_0_in_the_x_basis_forced_to_zero_leaves_ket_plus():
    tableau = create_n_ket0_state(1)
    outcome = measure_x(tableau, 0, measurement_determinism=0)
    assert outcome == 0
    assert np.allclose(tableau_to_density(tableau), _PLUS)


def test_measuring_ket_0_in_the_x_basis_forced_to_one_leaves_ket_minus():
    tableau = create_n_ket0_state(1)
    outcome = measure_x(tableau, 0, measurement_determinism=1)
    assert outcome == 1
    assert np.allclose(tableau_to_density(tableau), _MINUS)


def test_measuring_ket_0_in_the_y_basis_forced_to_zero_leaves_ket_plus_y():
    tableau = create_n_ket0_state(1)
    outcome = measure_y(tableau, 0, measurement_determinism=0)
    assert outcome == 0
    assert np.allclose(tableau_to_density(tableau), _PLUS_Y)


def test_measuring_ket_0_in_the_y_basis_forced_to_one_leaves_ket_minus_y():
    tableau = create_n_ket0_state(1)
    outcome = measure_y(tableau, 0, measurement_determinism=1)
    assert outcome == 1
    assert np.allclose(tableau_to_density(tableau), _MINUS_Y)


def _bell_state():
    tableau = create_n_ket0_state(2)
    tableau = hadamard_gate(tableau, 0)
    tableau = cnot_gate(tableau, 0, 1)
    return tableau


def test_measuring_one_qubit_of_a_bell_pair_in_the_x_basis_leaves_the_other_correlated():
    # Measuring qubit 0 of a Bell pair in the X basis is probabilistic; the
    # untouched qubit collapses to the same X eigenstate. Getting this right
    # depends on the measured qubit's row landing back in its own frame
    # before the surviving correlation is read off it.
    tableau = _bell_state()
    outcome = measure_x(tableau, 0, measurement_determinism=1)
    assert outcome == 1
    assert np.allclose(tableau_to_density(tableau), np.kron(_MINUS, _MINUS))
