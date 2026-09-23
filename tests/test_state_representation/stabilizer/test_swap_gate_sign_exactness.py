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
Conjugating by SWAP permutes qubit labels within each generator, so the
symplectic columns move -- but a sign belongs to the row it is on, not to
either of the columns that row happens to touch, and a swap does not permute
rows. ``swap_gate`` therefore moves the two columns and leaves both phase
vectors exactly where they are.

These tests build small hand-checkable states and check the swap against
the density matrix of the tableau's signed generators.
"""

import numpy as np

import graphiq.backends.state_rep_conversion as rc
from graphiq.backends.stabilizer.functions.clifford import (
    create_n_ket0_state,
    swap_gate,
)
from graphiq.backends.stabilizer.functions.transformation import (
    cnot_gate,
    hadamard_gate,
    x_gate,
)


def tableau_to_density(tableau):
    """The density matrix of a Clifford tableau's stabilizer half, signs included."""
    return rc.stabilizer_to_density(tableau.to_stabilizer())


def _expect(index, n_qubits):
    expected = np.zeros((2**n_qubits, 2**n_qubits))
    expected[index, index] = 1.0
    return expected


def test_swapping_ket_10_gives_ket_01():
    tableau = create_n_ket0_state(2)
    tableau = x_gate(tableau, 0)
    tableau = swap_gate(tableau, 0, 1)
    assert np.allclose(tableau_to_density(tableau), _expect(0b01, 2))


def test_swapping_two_qubits_of_an_entangled_three_qubit_state():
    # Swapping qubits 0 and 2 of a GHZ3 state carries a stabilizer
    # generator, ZIZ, whose sign must land on the swapped row, not drift
    # onto a different generator through a column-index-keyed permutation.
    tableau = create_n_ket0_state(3)
    tableau = hadamard_gate(tableau, 0)
    tableau = cnot_gate(tableau, 0, 1)
    tableau = cnot_gate(tableau, 0, 2)
    tableau = x_gate(tableau, 1)
    tableau = swap_gate(tableau, 0, 2)

    # H(0), CNOT(0,1), CNOT(0,2), X(1) gives (|010> + |101>) / sqrt(2); swapping
    # qubits 0 and 2 leaves it unchanged, since both terms are symmetric in them.
    expected = np.zeros((8, 8))
    for i in (0b010, 0b101):
        for j in (0b010, 0b101):
            expected[i, j] = 0.5

    assert np.allclose(tableau_to_density(tableau), expected)
