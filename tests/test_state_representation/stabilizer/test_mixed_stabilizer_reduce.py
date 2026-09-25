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
``MixedStabilizer.reduce`` folds identical tableaux in the mixture together,
summing their probabilities. Every match has to be found: comparing a
matched entry and moving on, without also re-checking the entry that shifted
into its place, would leave later duplicates unmerged.

These tests build mixtures out of copies of the very same tableau, so the
only thing under test is whether every match gets folded in. Folding
different tableaux of the same state (other destabilizer rows or signs, or
another generator order) is covered by
tests/test_noise/test_mixed_stabilizer_reduce.py.
"""

from graphiq.backends.stabilizer.clifford_tableau import CliffordTableau
from graphiq.backends.stabilizer.state import MixedStabilizer


def test_four_identical_tableaux_reduce_to_one():
    tableau = CliffordTableau(2)
    mixed = MixedStabilizer([(0.25, tableau.copy()) for _ in range(4)])

    mixed.reduce()

    assert len(mixed.mixture) == 1
    probability, reduced = mixed.mixture[0]
    assert probability == 1.0
    assert reduced == tableau


def test_reduce_still_keeps_distinct_tableaux_separate():
    zero_state = CliffordTableau(2)
    one_state = CliffordTableau(2)
    one_state.phase[one_state.n_qubits] = 1  # a different generator sign

    mixed = MixedStabilizer(
        [
            (0.5, zero_state.copy()),
            (0.5, one_state.copy()),
        ]
    )

    mixed.reduce()

    assert len(mixed.mixture) == 2


def test_reduce_folds_three_matching_entries_in_a_row():
    # A run of three matching entries is exactly where an off-by-one skip
    # would drop the third instead of merging it: once the first match is
    # removed, the next matching entry shifts into the position such a skip
    # would pass over.
    tableau = CliffordTableau(2)
    other = CliffordTableau(2)
    other.phase[other.n_qubits] = 1

    mixed = MixedStabilizer(
        [
            (0.25, tableau.copy()),
            (0.25, tableau.copy()),
            (0.25, tableau.copy()),
            (0.25, other.copy()),
        ]
    )

    mixed.reduce()

    assert len(mixed.mixture) == 2
    probabilities = sorted(p for p, _ in mixed.mixture)
    assert probabilities == [0.25, 0.75]
