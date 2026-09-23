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
``Stabilizer.apply_x_measurement`` measures a qubit in the X basis and must
leave the tableau in a valid post-measurement state expressed in the qubit's
own (unrotated) basis -- not in the Z frame the measurement is taken in.

These tests build small states through the ``Stabilizer`` class's own public
gate methods and check the outcome and the density matrix of the resulting
tableau. Every case forces its outcome with ``measurement_determinism`` so
nothing here depends on the unseeded global RNG.
"""

import numpy as np

import graphiq.backends.state_rep_conversion as rc
from graphiq.backends.stabilizer.state import Stabilizer


def tableau_to_density(tableau):
    """The density matrix of a Clifford tableau's stabilizer half, signs included."""
    return rc.stabilizer_to_density(tableau.to_stabilizer())


_PLUS = np.array([[0.5, 0.5], [0.5, 0.5]])
_MINUS = np.array([[0.5, -0.5], [-0.5, 0.5]])


def test_measuring_ket_plus_in_the_x_basis_returns_zero_and_leaves_it_unchanged():
    stab = Stabilizer(1)
    stab.apply_hadamard(0)

    outcome = stab.apply_x_measurement(0, measurement_determinism=0)

    assert outcome == 0
    assert np.allclose(tableau_to_density(stab.tableau), _PLUS)


def test_measuring_ket_0_in_the_x_basis_forced_to_zero_leaves_ket_plus():
    stab = Stabilizer(1)

    outcome = stab.apply_x_measurement(0, measurement_determinism=0)

    assert outcome == 0
    assert np.allclose(tableau_to_density(stab.tableau), _PLUS)


def test_measuring_ket_0_in_the_x_basis_forced_to_one_leaves_ket_minus():
    stab = Stabilizer(1)

    outcome = stab.apply_x_measurement(0, measurement_determinism=1)

    assert outcome == 1
    assert np.allclose(tableau_to_density(stab.tableau), _MINUS)


def test_measuring_one_qubit_of_a_bell_pair_in_the_x_basis_leaves_the_other_correlated():
    # Measuring qubit 0 of a Bell pair in the X basis is probabilistic; the
    # untouched qubit collapses to the same X eigenstate. Getting this right
    # depends on the measured qubit's row landing back in its own frame,
    # not the Z frame the measurement is taken in, before the surviving
    # correlation is read off it.
    stab = Stabilizer(2)
    stab.apply_hadamard(0)
    stab.apply_cnot(0, 1)

    outcome = stab.apply_x_measurement(0, measurement_determinism=1)

    assert outcome == 1
    assert np.allclose(tableau_to_density(stab.tableau), np.kron(_MINUS, _MINUS))


def test_measuring_qubit_one_of_an_asymmetric_state_measures_that_qubit():
    # The Bell case above cannot see which qubit was measured: that state is
    # symmetric under exchanging the two, so measuring either gives the same
    # answer. H then CNOT then H on qubit 1 breaks the symmetry -- qubit 0 is
    # so this pins that the position argument reaches the measurement rather
    # than the method always measuring qubit 0.
    #
    # H, CNOT, H leaves (|0>|+> + |1>|->)/sqrt(2). Forcing qubit 1 onto the -1
    # eigenstate of X selects the second branch, so the state collapses to
    # |1>|->. Measuring qubit 0 instead would leave both qubits in an X
    # eigenstate and never produce a |1> on qubit 0.
    stab = Stabilizer(2)
    stab.apply_hadamard(0)
    stab.apply_cnot(0, 1)
    stab.apply_hadamard(1)

    outcome = stab.apply_x_measurement(1, measurement_determinism=1)

    ket_one = np.array([[0.0, 0.0], [0.0, 1.0]])
    assert outcome == 1
    assert np.allclose(tableau_to_density(stab.tableau), np.kron(ket_one, _MINUS))


def test_forcing_an_outcome_cannot_override_a_determined_one():
    # measurement_determinism selects among genuinely possible outcomes; it
    # does not override a determined one. |-> is the -1 eigenstate of X, so
    # measuring it in the X basis yields 1 whatever is asked for.
    for determinism in (0, 1, "probabilistic"):
        stab = Stabilizer(1)
        stab.apply_hadamard(0)
        stab.apply_sigmaz(0)

        outcome = stab.apply_x_measurement(0, measurement_determinism=determinism)

        assert outcome == 1, f"determinism={determinism}"
        assert np.allclose(tableau_to_density(stab.tableau), _MINUS)
