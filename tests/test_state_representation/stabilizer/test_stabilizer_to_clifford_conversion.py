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
Building a ``CliffordTableau`` from a ``StabilizerTableau`` finds the
destabilizers by reducing the state to :math:`|0\\rangle^{\\otimes n}` and
replaying the reducing circuit backwards. The reduction runs in place, so the
conversion works on a copy: the caller's tableau is an input, and it still
holds the state it held before. Solvers hand their working tableau to this
conversion, so a conversion that reduced the caller's own generators would
leave them holding the all-zero state instead.

The conversion also has to preserve the state, not just leave the argument
alone -- the Clifford tableau's stabilizer half generates the same group the
stabilizer tableau did, signs included.
"""

import numpy as np

from graphiq.backends.stabilizer.clifford_tableau import CliffordTableau
from graphiq.backends.stabilizer.functions.rep_conversion import (
    clifford_from_stabilizer,
    stabilizer_from_clifford,
)
from graphiq.backends.stabilizer.functions.stabilizer import canonical_form
from graphiq.backends.stabilizer.tableau import StabilizerTableau


def _signed_ghz3():
    """A GHZ3 state whose generators carry signs, so a lost phase shows up."""
    tableau = StabilizerTableau(3)
    tableau.from_labels(["XXX", "ZZI", "IZZ"])
    tableau.phase = np.array([1, 0, 1])
    return tableau


def test_the_clifford_tableau_constructor_leaves_the_stabilizer_tableau_alone():
    state = _signed_ghz3()
    before = state.copy()

    CliffordTableau(state)

    assert state == before


def test_clifford_from_stabilizer_leaves_its_argument_alone():
    state = _signed_ghz3()
    before = state.copy()

    clifford_from_stabilizer(state)

    assert state == before


def test_the_clifford_tableau_holds_the_state_it_was_built_from():
    state = _signed_ghz3()
    # Taken before the conversion so that this holds the conversion to the
    # state, independently of whether it leaves the argument alone.
    expected = canonical_form(state.copy())

    recovered = stabilizer_from_clifford(CliffordTableau(state))

    assert canonical_form(recovered) == expected
