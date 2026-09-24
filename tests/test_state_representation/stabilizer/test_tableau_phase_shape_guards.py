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
Shape guards on the phase and i-phase vectors of both tableau classes.

The constructors used to replace a phase of the wrong length, or a plain list,
with zeros, so a requested -1 sign silently became +1. They accepted a column-shaped
phase as is, which later gate updates broadcast into an (n, n) array. A 0-d phase
raised a bare IndexError, and CliffordTableau dropped a phase passed alongside a
tableau to copy. The phase setters and ``_reset`` asserted only the length, and
``__eq__`` compared phases by broadcasting. These tests pin the ``ValueError`` each
path now raises, and the shape-strict comparison.
"""

import warnings

import numpy as np
import pytest

from graphiq.backends.stabilizer.tableau import StabilizerTableau
from graphiq.backends.stabilizer.clifford_tableau import CliffordTableau


def test_stabilizer_tableau_rejects_column_phase():
    with pytest.raises(ValueError):
        StabilizerTableau(3, np.ones((3, 1)))


def test_clifford_tableau_rejects_column_phase():
    with pytest.raises(ValueError):
        CliffordTableau(2, np.ones((4, 1)))


def test_stabilizer_tableau_phase_setter_rejects_column_phase():
    tableau = StabilizerTableau(3)
    with pytest.raises(ValueError):
        tableau.phase = np.ones((3, 1))


def test_clifford_tableau_phase_setter_rejects_column_phase():
    tableau = CliffordTableau(2)
    with pytest.raises(ValueError):
        tableau.phase = np.ones((4, 1))


def test_stabilizer_tableau_rejects_wrong_length_phase():
    with pytest.raises(ValueError):
        StabilizerTableau(3, np.array([1, 1]))


def test_clifford_tableau_rejects_wrong_length_phase():
    with pytest.raises(ValueError):
        CliffordTableau(2, np.array([1, 1]))


def test_stabilizer_tableau_rejects_scalar_phase_with_a_named_error():
    # A 0-d phase used to reach shape[0] and raise a bare IndexError; this
    # pins that it is now the same named ValueError as any other bad shape.
    with pytest.raises(ValueError):
        StabilizerTableau(3, np.array(1))


def test_phase_none_still_yields_zeros():
    assert np.array_equal(StabilizerTableau(3).phase, np.zeros(3))
    assert np.array_equal(StabilizerTableau(3, phase=None).phase, np.zeros(3))
    assert np.array_equal(CliffordTableau(2).phase, np.zeros(4))
    assert np.array_equal(CliffordTableau(2, phase=None).phase, np.zeros(4))


def test_a_python_list_phase_is_accepted():
    assert np.array_equal(StabilizerTableau(3, [1, 0, 1]).phase, [1, 0, 1])
    assert np.array_equal(CliffordTableau(2, [1, 0, 1, 0]).phase, [1, 0, 1, 0])


def test_clifford_tableau_from_clifford_tableau_rejects_a_phase_argument():
    src = CliffordTableau(2)
    src.phase = np.array([1, 0, 1, 0])
    passed_phase = np.array([0, 1, 0, 1])

    with pytest.raises(ValueError):
        CliffordTableau(src, passed_phase)
    with pytest.raises(ValueError):
        CliffordTableau(src, phase=passed_phase)


def test_clifford_tableau_from_stabilizer_tableau_rejects_a_phase_argument():
    src = StabilizerTableau(2)
    passed_phase = np.array([0, 1, 0, 1])

    with pytest.raises(ValueError):
        CliffordTableau(src, passed_phase)
    with pytest.raises(ValueError):
        CliffordTableau(src, phase=passed_phase)


def test_iphase_setter_rejects_a_column_vector():
    tableau = CliffordTableau(2)
    with pytest.raises(ValueError):
        tableau.iphase = np.ones((4, 1))


def test_stabilizer_tableau_expand_rejects_a_column_new_phase():
    tableau = StabilizerTableau(2)
    new_table = np.hstack([np.zeros((3, 3)), np.eye(3)])
    with pytest.raises(ValueError):
        tableau.expand(new_table, np.ones((3, 1)))


def test_stabilizer_tableau_shrink_rejects_a_column_new_phase():
    tableau = StabilizerTableau(3)
    new_table = np.hstack([np.zeros((2, 2)), np.eye(2)])
    with pytest.raises(ValueError):
        tableau.shrink(new_table, np.ones((2, 1)))


def test_clifford_tableau_expand_rejects_a_column_new_phase():
    tableau = CliffordTableau(2)
    new_table = np.eye(6)
    with pytest.raises(ValueError):
        tableau.expand(new_table, np.ones((6, 1)), np.zeros(6))


def test_clifford_tableau_shrink_rejects_a_column_new_phase():
    tableau = CliffordTableau(3)
    new_table = np.eye(4)
    with pytest.raises(ValueError):
        tableau.shrink(new_table, np.ones((4, 1)), np.zeros(4))


def test_clifford_tableau_expand_rejects_a_wrong_length_new_iphase():
    tableau = CliffordTableau(2)
    new_table = np.eye(6)
    with pytest.raises(ValueError):
        tableau.expand(new_table, np.zeros(6), np.zeros(3))


def test_stabilizer_tableau_eq_reports_equal_phases_as_equal():
    a = StabilizerTableau(3, np.array([1, 0, 1]))
    b = StabilizerTableau(3, np.array([1, 0, 1]))

    assert a == b


def test_stabilizer_tableau_eq_reports_unequal_phases_as_unequal():
    a = StabilizerTableau(3, np.array([1, 0, 1]))
    b = StabilizerTableau(3, np.array([0, 0, 0]))

    assert a != b


def test_stabilizer_tableau_eq_does_not_broadcast_a_column_phase_into_a_match():
    # The constructor and setter now reject a column phase outright, but the
    # phase getter returns _phase by identity, so `t.phase.shape = (n, 1)`
    # reshapes one in place past both guards. A same-valued column used to
    # compare equal to a row by broadcasting the comparison into an (n, n)
    # matrix; the shape-strict comparison must report them as unequal.
    row = StabilizerTableau(3, np.ones(3))
    column = StabilizerTableau(3, np.ones(3))
    column.phase.shape = (3, 1)

    assert row != column


def test_clifford_tableau_eq_reports_equal_phase_and_iphase_as_equal():
    a = CliffordTableau(2)
    a.phase = np.array([1, 0, 1, 0])
    a.iphase = np.array([0, 1, 0, 1])

    b = CliffordTableau(2)
    b.phase = np.array([1, 0, 1, 0])
    b.iphase = np.array([0, 1, 0, 1])

    assert a == b


def test_clifford_tableau_eq_reports_unequal_iphase_as_unequal():
    a = CliffordTableau(2)
    a.iphase = np.array([1, 0, 0, 0])

    b = CliffordTableau(2)

    assert a != b


def test_clifford_tableau_eq_does_not_broadcast_a_column_iphase_into_a_match():
    # Same in-place reshape as above, on the i-phase vector.
    a = CliffordTableau(2)
    a.iphase = np.ones(4)

    b = CliffordTableau(2)
    b.iphase = np.ones(4)
    b.iphase.shape = (4, 1)

    assert a != b


def test_stabilizer_tableau_eq_of_different_sizes_is_false_without_a_warning():
    # Comparing phases of different lengths with == is an elementwise comparison
    # numpy cannot do; it warns and falls back to a scalar. The shape-strict
    # comparison just reports the tableaux as unequal.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert StabilizerTableau(2) != StabilizerTableau(3)


def test_clifford_tableau_eq_of_different_sizes_is_false_without_a_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert CliffordTableau(2) != CliffordTableau(3)
