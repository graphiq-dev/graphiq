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
``insert_qubit`` builds the enlarged phase and i-phase vectors with
``np.insert``, whose index list is read against the *original* array, so the
new stabilizer row goes at ``n_qubits + new_position``. That holds at either
end as well as in the middle: appending, which is what ``add_qubit`` does,
indexes the last slot of the original vector rather than one past it.

These tests build small states that carry a non-zero phase and check the
inserted qubit's position, and the untouched qubits' signs, against the density matrix of the tableau's signed generators.
"""

import numpy as np

import graphiq.backends.state_rep_conversion as rc
from graphiq.backends.stabilizer.functions.clifford import (
    add_qubit,
    create_n_ket0_state,
    insert_qubit,
)
from graphiq.backends.stabilizer.functions.transformation import (

    cnot_gate,
    hadamard_gate,
    x_gate,
)


def tableau_to_density(tableau):
    """The density matrix of a Clifford tableau's stabilizer half, signs included."""
    return rc.stabilizer_to_density(tableau.to_stabilizer())


def _ket_01():
    tableau = create_n_ket0_state(2)
    tableau = x_gate(tableau, 1)
    return tableau


def _bell_state():
    tableau = create_n_ket0_state(2)
    tableau = hadamard_gate(tableau, 0)
    tableau = cnot_gate(tableau, 0, 1)
    return tableau


def _expect(index, n_qubits):
    expected = np.zeros((2**n_qubits, 2**n_qubits))
    expected[index, index] = 1.0
    return expected


def test_appending_an_isolated_qubit_to_a_state_with_a_nonzero_phase():
    # add_qubit is insert_qubit at new_position == n_qubits, the append case:
    # the index is read against the original vector, so the new stabilizer
    # row's slot is its last, not one past its end. add_qubit rests entirely
    # on this, which is why it gets its own test.
    tableau = add_qubit(_ket_01())
    assert np.allclose(tableau_to_density(tableau), _expect(0b010, 3))


def test_inserting_a_qubit_in_the_middle_of_a_state_with_a_nonzero_phase():
    # Inserting at position 1 into |01> shifts the old qubit 1 to position 2;
    # the inserted row's sign lands with its own qubit, not one to the right.
    tableau = insert_qubit(_ket_01(), 1)
    assert np.allclose(tableau_to_density(tableau), _expect(0b001, 3))


def test_inserting_a_qubit_at_the_front_of_an_entangled_state():
    tableau = insert_qubit(_bell_state(), 0)
    expected = np.kron(_expect(0, 1), tableau_to_density(_bell_state()))
    assert np.allclose(tableau_to_density(tableau), expected)
