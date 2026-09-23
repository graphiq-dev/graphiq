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
The generator-string view of a Clifford tableau.

``stabilizer_to_labels`` and ``destabilizer_to_labels`` render half the
tableau as Pauli strings; ``stabilizer_from_labels`` and
``destabilizer_from_labels`` set that half from them. The strings carry no
sign -- the phase and i-phase vectors are separate and the setters leave them
alone -- so a label round trip preserves the symplectic half only, and a
caller restoring a signed state has to carry the phase itself.
"""

import numpy as np

from graphiq.backends.stabilizer.clifford_tableau import CliffordTableau


def test_a_fresh_tableau_renders_as_the_all_zero_state():
    # |00> is stabilized by ZI and IZ, with XI and IX as the destabilizers
    # pairing with them by row index.
    tableau = CliffordTableau(2)

    assert tableau.destabilizer_to_labels() == ["XI", "IX"]
    assert tableau.stabilizer_to_labels() == ["ZI", "IZ"]


def test_labels_round_trip_through_the_setters():
    tableau = CliffordTableau(2)

    tableau.stabilizer_from_labels(["XX", "ZZ"])
    tableau.destabilizer_from_labels(["ZI", "IX"])

    assert tableau.stabilizer_to_labels() == ["XX", "ZZ"]
    assert tableau.destabilizer_to_labels() == ["ZI", "IX"]


def test_the_setters_write_only_their_own_half():
    tableau = CliffordTableau(2)
    destabilizers_before = tableau.destabilizer_to_labels()

    tableau.stabilizer_from_labels(["XX", "ZZ"])

    assert tableau.destabilizer_to_labels() == destabilizers_before


def test_the_setters_leave_the_phase_vectors_alone():
    # The strings carry no sign, so the setters cannot restore one. A caller
    # rebuilding a signed state has to set the phase itself; this pins that
    # the setters do not quietly zero or invent one.
    tableau = CliffordTableau(2)
    tableau.phase[2] = 1
    tableau.iphase[0] = 1

    tableau.stabilizer_from_labels(["XX", "ZZ"])
    tableau.destabilizer_from_labels(["ZI", "IX"])

    assert np.array_equal(tableau.phase, [0, 0, 1, 0])
    assert np.array_equal(tableau.iphase, [1, 0, 0, 0])


def test_a_tableau_renders_both_halves_and_both_phases():
    # __str__ is the only caller of the label getters in the library, so this
    # is what keeps that path from breaking unnoticed.
    rendered = str(CliffordTableau(1))

    assert "Destabilizers" in rendered
    assert "Stabilizer" in rendered
    assert "['X']" in rendered
    assert "['Z']" in rendered
