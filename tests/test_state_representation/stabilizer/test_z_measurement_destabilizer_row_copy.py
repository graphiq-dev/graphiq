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
When a probabilistic Z-basis measurement replaces a stabilizer generator with
the fresh outcome, the generator it displaces has to move somewhere: it
becomes the new destabilizer partner for the measured qubit. Aaronson-Gottesman
say to set that whole destabilizer row equal to the displaced generator's row,
sign included, and the code does: the destabilizer row's phase and i-phase
always end up matching the displaced generator's own, never whatever those
entries happened to hold going into the measurement (typically leftover
arithmetic from combining that row with others).

These tests build tableaux by hand so the displaced generator's phase and
i-phase are known quantities, and check that the destabilizer row ends up
carrying exactly them -- copying only the symplectic (x/z) bits and leaving
the sign behind would fail every one of them.
"""

import numpy as np

from graphiq.backends.stabilizer.clifford_tableau import CliffordTableau
from graphiq.backends.stabilizer.functions.clifford import (
    create_n_ket0_state,
    z_measurement_gate,
)
from graphiq.backends.stabilizer.functions.transformation import (
    cnot_gate,
    hadamard_gate,
)


def _bell_state():
    tableau = create_n_ket0_state(2)
    tableau = hadamard_gate(tableau, 0)
    tableau = cnot_gate(tableau, 0, 1)
    return tableau


def test_the_displaced_generators_sign_moves_to_the_destabilizer_row():
    # (|00> - |11>) / sqrt(2): stabilizers -XX, +ZZ. Measuring qubit 0 finds
    # a single stabilizer row with x-support there (the XX row), so the
    # row-sum loop above the copy never touches it -- its phase is exactly
    # what it was before the measurement, a sign of 1. A destabilizer row
    # that fails to inherit it stays at its initial 0.
    tableau = _bell_state()
    tableau.phase[2] = 1  # the XX stabilizer row becomes "-XX"

    measured, outcome, x_p = z_measurement_gate(tableau, 0, measurement_determinism=0)

    assert x_p == 2
    assert measured.phase[x_p - measured.n_qubits] == 1


def test_the_displaced_generators_i_phase_moves_to_the_destabilizer_row():
    # A hand-built Bell state in a basis where the destabilizer partner of
    # the measured stabilizer row shares its x-support: x-column [1,0,1,0]
    # for qubit 0 puts both the destabilizer row and the stabilizer row (XX)
    # in the row-sum loop, which combines them and leaves the destabilizer
    # row's i-phase at whatever that combination produced -- 1 here -- unless
    # the copy overwrites it with the displaced generator's own i-phase, 0.
    table = np.array(
        [
            [1, 1, 1, 0],  # destabilizer 0: YX
            [0, 1, 0, 0],  # destabilizer 1: IX
            [1, 1, 0, 0],  # stabilizer 0: XX
            [0, 0, 1, 1],  # stabilizer 1: ZZ
        ]
    )
    tableau = CliffordTableau(table)

    measured, outcome, x_p = z_measurement_gate(tableau, 0, measurement_determinism=0)

    assert x_p == 2
    assert measured.iphase[x_p - measured.n_qubits] == 0


def test_the_fresh_generator_row_carries_no_i_phase():
    # The measured row is overwritten with a fresh +/-Z_a generator, which has
    # a bare +1/-1 sign and no factor of i. Its i-phase entry should read 0
    # regardless of what it held going into the measurement.
    tableau = _bell_state()
    tableau.iphase[2] = 1  # a value the row's own bookkeeping should not keep

    measured, outcome, x_p = z_measurement_gate(tableau, 0, measurement_determinism=0)

    assert x_p == 2
    assert measured.iphase[x_p] == 0
