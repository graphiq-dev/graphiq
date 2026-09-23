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
"""
``inverse_circuit`` reduces a stabilizer state to
:math:`|0\\rangle^{\\otimes n}` and hands back the gates that do it. Its
Hadamard block is what makes the reduction possible: the blocks after it clear
the x matrix by column operations, which needs a pivot on every diagonal entry
of that matrix, and only Hadamards can create one in a column whose generators
are all Z.

The guarantee those blocks rely on, and which these tests pin down, is that the
Hadamard block leaves the x matrix with a pivot in every column. The state
below is a case where a per-column walk that parks one generator per column
without checking that the generator has support there runs out of generators
before the last column.

The reduction is silent -- nothing in ``inverse_circuit`` raises when it fails
-- so these tests assert on the returned tableau directly rather than only
through a reference. A state that did not reduce still carries X or Y
generators, which needs no oracle to spot.
"""

import numpy as np

from graphiq.backends.stabilizer.functions.stabilizer import (
    canonical_form,
    inverse_circuit,
)
from graphiq.backends.stabilizer.functions.transformation import run_circuit
from graphiq.backends.stabilizer.tableau import StabilizerTableau


def _stabilizer_state(labels, phase):
    tableau = StabilizerTableau(len(labels))
    tableau.from_labels(labels)
    tableau.phase = np.array(phase)
    return tableau


#: A five-qubit state written in canonical form, chosen so that qubit 1 is
#: stabilized on its own by ``IZIII`` while every other generator's support
#: sits in columns 0, 2, 3 and 4. A per-column walk hands column 1 the lone
#: ``IZIII``, which needs no Hadamard, and then has nothing left with support
#: in column 4 by the time it gets there.
_NO_PIVOT_IN_THE_LAST_COLUMN = (
    ["XIYZZ", "IIZYZ", "IIZZY", "ZIZII", "IZIII"],
    [0, 1, 1, 0, 1],
)


def test_a_state_with_a_lone_z_generator_reduces_to_the_all_zero_state():
    state = _stabilizer_state(*_NO_PIVOT_IN_THE_LAST_COLUMN)

    reduced, _ = inverse_circuit(state.copy())

    assert reduced == StabilizerTableau(5)


def test_the_circuit_for_a_state_with_a_lone_z_generator_zeroes_it():
    state = _stabilizer_state(*_NO_PIVOT_IN_THE_LAST_COLUMN)

    _, circuit_list = inverse_circuit(state.copy())

    # The circuit is compared in canonical form because it is only required to
    # land the state on |0...0>, not to reproduce any particular generators
    # for it: the reduction interleaves row operations that never enter the
    # circuit, so the two end up in different bases of the same group.
    assert canonical_form(run_circuit(state.copy(), list(circuit_list))) == (
        StabilizerTableau(5)
    )


def test_reducing_the_all_zero_state_asks_for_no_gates():
    reduced, circuit_list = inverse_circuit(StabilizerTableau(4))

    assert circuit_list == []
    assert reduced == StabilizerTableau(4)


def test_the_reducing_circuit_skips_the_qubits_that_are_already_reduced():
    # A Bell pair on qubits 0 and 2, with qubits 1 and 3 in |0> and |1>. A
    # generator that is a bare Z on one qubit says that qubit is unentangled
    # and in |0> or |1>, so no gate but the sign fix belongs on it.
    state = _stabilizer_state(
        ["XIXI", "ZIZI", "IZII", "IIIZ"],
        [0, 0, 0, 1],
    )

    reduced, circuit_list = inverse_circuit(state.copy())

    assert reduced == StabilizerTableau(4)
    assert [gate for gate in circuit_list if set(gate[1:]) & {1, 3}] == [("X", 3)]
