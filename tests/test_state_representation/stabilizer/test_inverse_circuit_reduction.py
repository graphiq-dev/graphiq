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
:math:`|0\\rangle^{\\otimes n}` and hands back the gates that do it. The
blocks after its opening Hadamard block clear the x matrix by column operations,
which needs a pivot on every diagonal entry of that matrix, and only Hadamards
can create one in a column whose generators are all Z.

It first reduces with a Hadamard block that does not always leave such a pivot,
and keeps that result only if it is exactly the standard all-zero tableau.
Otherwise it starts again from the input with a block that places a pivot on
every column, and raises if that also misses. The states below include one where
the first block runs out of generators before the last column, and one where it
reaches the all-zero state written with other Z generators.

These tests assert on the returned tableau directly, by exact equality with the
standard all-zero tableau, which needs no reference circuit.
"""

import numpy as np

from graphiq.backends.stabilizer.functions.metric import fidelity
from graphiq.backends.stabilizer.functions.rep_conversion import (
    clifford_from_stabilizer,
)
from graphiq.backends.stabilizer.functions.stabilizer import (
    canonical_form,
    inverse_circuit,
)
from graphiq.backends.stabilizer.functions.transformation import (
    control_z_gate,
    cnot_gate,
    hadamard_gate,
    phase_gate,
    run_circuit,
    x_gate,
    z_gate,
)
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
    # A Bell pair on qubits 0 and 2, with qubits 1 and 3 in |0> and |1>.
    # Qubits 1 and 3 are unentangled and already in a computational basis
    # state, so no gate but the sign fix belongs on them.
    state = _stabilizer_state(
        ["XIXI", "ZIZI", "IZII", "IIIZ"],
        [0, 0, 0, 1],
    )

    reduced, circuit_list = inverse_circuit(state.copy())

    assert reduced == StabilizerTableau(4)
    assert [gate for gate in circuit_list if set(gate[1:]) & {1, 3}] == [("X", 3)]


#: The tableau TimeReversedSolver hands to ``inverse_circuit`` to disentangle
#: the emitters, for one random six-qubit case (four photons, two emitters).
#: The first Hadamard block reduces it, so its circuit is the one that block
#: gives.
_EMITTER_STAGE = (
    ["ZIIIII", "IZIIII", "IIZIII", "IIIZII", "IIIIYX", "IIIIZZ"],
    [0, 0, 0, 0, 0, 0],
)


def test_a_state_the_first_hadamard_block_reduces_keeps_its_circuit():
    state = _stabilizer_state(*_EMITTER_STAGE)

    reduced, circuit_list = inverse_circuit(state.copy())

    assert circuit_list == [
        ("CNOT", 4, 5),
        ("CZ", 4, 5),
        ("P", 4),
        ("H", 4),
        ("X", 4),
    ]
    assert reduced == StabilizerTableau(6)


#: A nine-qubit state that the first Hadamard block reduces to |0...0>, but
#: with a generator that is ``Z_7 Z_8`` rather than ``Z_7``. The state is
#: right, but the inner product in the stabilizer metric reads the returned
#: generators row by row and needs each to be the bare ``Z_j``.
_ZERO_STATE_WRITTEN_WITH_OTHER_GENERATORS = (
    [
        "IXIXIZIIX",
        "YXZXIIYXX",
        "IIZIZIIII",
        "IZIZIIIII",
        "YXYXXIYZX",
        "YYYXXYYXX",
        "XXZXIIXXX",
        "IIZIIZIXI",
        "IZZIZIIIZ",
    ],
    [1, 1, 1, 0, 0, 1, 0, 0, 1],
)


def test_the_reduced_tableau_is_the_standard_all_zero_tableau():
    state = _stabilizer_state(*_ZERO_STATE_WRITTEN_WITH_OTHER_GENERATORS)

    reduced, circuit_list = inverse_circuit(state.copy())

    assert reduced == StabilizerTableau(9)
    assert canonical_form(run_circuit(state.copy(), list(circuit_list))) == (
        StabilizerTableau(9)
    )


def test_fidelity_tells_a_state_from_one_a_single_x_makes_orthogonal():
    labels, phase = _ZERO_STATE_WRITTEN_WITH_OTHER_GENERATORS
    state = clifford_from_stabilizer(_stabilizer_state(labels, phase))

    # X on qubit 2 anticommutes with a generator, so it gives an orthogonal
    # state.
    flipped = x_gate(state.copy(), 2)
    # Generator 0 stabilizes the state, so applying it gives the same state.
    same = state.copy()
    for qubit, pauli in enumerate(labels[0]):
        if pauli in "XY":
            same = x_gate(same, qubit)
        if pauli in "ZY":
            same = z_gate(same, qubit)

    assert fidelity(state, flipped) == 0
    assert fidelity(state, same) == 1


def _random_stabilizer_state(rng, n_qubits, n_gates):
    tableau = StabilizerTableau(n_qubits)
    for _ in range(n_gates):
        gate = rng.integers(5)
        if gate < 3:
            qubit = int(rng.integers(n_qubits))
            tableau = (hadamard_gate, phase_gate, x_gate)[gate](tableau, qubit)
        else:
            control, target = (int(q) for q in rng.choice(n_qubits, 2, replace=False))
            tableau = (cnot_gate, control_z_gate)[gate - 3](tableau, control, target)
    return tableau


def _states_to_reduce():
    yield _stabilizer_state(*_NO_PIVOT_IN_THE_LAST_COLUMN)
    yield _stabilizer_state(*_ZERO_STATE_WRITTEN_WITH_OTHER_GENERATORS)
    yield _stabilizer_state(*_EMITTER_STAGE)
    rng = np.random.default_rng(0)
    for _ in range(200):
        yield _random_stabilizer_state(
            rng, int(rng.integers(2, 8)), int(rng.choice([5, 20, 60]))
        )


def test_every_state_reduces_in_place_to_the_standard_all_zero_tableau():
    for state in _states_to_reduce():
        n_qubits = state.n_qubits
        argument = state.copy()

        reduced, circuit_list = inverse_circuit(argument)

        assert reduced is argument
        assert reduced == StabilizerTableau(n_qubits)
        assert canonical_form(run_circuit(state.copy(), list(circuit_list))) == (
            StabilizerTableau(n_qubits)
        )
