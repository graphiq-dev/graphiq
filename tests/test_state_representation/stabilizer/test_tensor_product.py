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
``tensor`` returns a fresh tableau holding the block-diagonal product of its
arguments, leaving every argument untouched. The destabilizer blocks, the
stabilizer blocks and both phase vectors are assembled in factor order, with
all the destabilizer phases ahead of all the stabilizer phases.

These tests check the product against the density matrix of the tableau's signed
generators,
confirm the inputs are read-only, and measure the result so that the
destabilizer half -- which that density matrix does not read -- is covered too.
"""

import numpy as np

import graphiq.backends.state_rep_conversion as rc
from graphiq.backends.stabilizer.functions.clifford import (
    create_n_ket0_state,
    create_n_ket1_state,
    tensor,
)
from graphiq.backends.stabilizer.functions.transformation import (

    z_gate,
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


def _expect(index, n_qubits):
    expected = np.zeros((2**n_qubits, 2**n_qubits))
    expected[index, index] = 1.0
    return expected


def test_tensor_of_ket_0_and_ket_1_is_ket_01():
    tableau = tensor([create_n_ket0_state(1), create_n_ket1_state(1)])
    assert np.allclose(tableau_to_density(tableau), _expect(0b01, 2))


def test_tensor_carries_the_sign_of_each_factor():
    # ket1 carries phase [0, 1] -- the sign the aliasing-free rebuild has to
    # place on the right qubit's row, not the left one's.
    tableau = tensor([create_n_ket1_state(1), create_n_ket0_state(1)])
    assert np.allclose(tableau_to_density(tableau), _expect(0b10, 2))


def test_tensor_of_three_factors_including_an_entangled_one():
    tableau = tensor([create_n_ket1_state(1), _bell_state(), create_n_ket0_state(1)])
    expected = np.kron(
        np.kron(_expect(1, 1), tableau_to_density(_bell_state())),
        _expect(0, 1),
    )
    assert np.allclose(tableau_to_density(tableau), expected)


def test_tensor_lays_out_the_destabilizer_half_and_the_phases():
    # That density matrix reads the stabilizer half and the stabilizer phases only, so
    # every test above is blind to the destabilizer blocks and to the
    # destabilizer phases. This pins both against a hand-worked expectation.
    #
    # Z|0> is one qubit whose destabilizer is -X and whose stabilizer is +Z, so
    # it is the smaller factor AND it carries the only non-zero phase, in the
    # destabilizer half. The Bell pair's generators are ZI, IX (destabilizers)
    # and XX, ZZ (stabilizers), all with a + sign. Tensored in that order the
    # three qubits therefore read -XII, IZI, IIX / +ZII, IXX, IZZ.
    tableau = tensor([z_gate(create_n_ket0_state(1), 0), _bell_state()])

    assert np.array_equal(tableau.destabilizer_x, [[1, 0, 0], [0, 0, 0], [0, 0, 1]])
    assert np.array_equal(tableau.destabilizer_z, [[0, 0, 0], [0, 1, 0], [0, 0, 0]])
    assert np.array_equal(tableau.stabilizer_x, [[0, 0, 0], [0, 1, 1], [0, 0, 0]])
    assert np.array_equal(tableau.stabilizer_z, [[1, 0, 0], [0, 0, 0], [0, 1, 1]])
    # Every destabilizer phase ahead of every stabilizer phase, factor by
    # factor -- not the two halves of each factor kept together.
    assert np.array_equal(tableau.phase, [1, 0, 0, 0, 0, 0])
    assert not np.any(tableau.iphase)


def test_tensor_does_not_mutate_its_arguments():
    first, second = create_n_ket0_state(1), create_n_ket1_state(1)
    first_before, second_before = first.copy(), second.copy()

    tensor([first, second])

    assert first == first_before
    assert second == second_before
