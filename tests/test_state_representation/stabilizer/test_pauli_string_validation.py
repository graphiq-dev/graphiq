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
``string_to_symplectic`` and ``get_stabilizer_element_by_string`` used to accept any
character, mapping everything that was not x/y/z (case-insensitively) to the identity
instead of rejecting it. A sign character, a stray space, a trailing newline, or a typo
would silently read as an extra identity qubit rather than raising. A ragged list of
labels was read at the width of the first label, so it was silently truncated when the
first label was the shortest and raised IndexError otherwise. A bare string was read as
one single-qubit generator per character, and an empty list raised IndexError. All of
these now raise ValueError.
"""

import numpy as np
import pytest

import graphiq.backends.stabilizer.functions.utils as sfu
import graphiq.backends.state_rep_conversion as rc
from graphiq.backends.stabilizer.tableau import StabilizerTableau


def test_string_to_symplectic_accepts_clean_input():
    x_matrix, z_matrix = sfu.string_to_symplectic(["ZI", "IZ"])

    assert np.array_equal(x_matrix, np.zeros((2, 2)))
    assert np.array_equal(z_matrix, np.array([[1, 0], [0, 1]]))


def test_string_to_symplectic_lowercase_matches_uppercase():
    lower_x, lower_z = sfu.string_to_symplectic(["zi", "iz"])
    upper_x, upper_z = sfu.string_to_symplectic(["ZI", "IZ"])

    assert np.array_equal(lower_x, upper_x)
    assert np.array_equal(lower_z, upper_z)


def test_string_to_symplectic_rejects_non_pauli_character():
    # The critical mutant: same width as a clean label, so only a character check (not a
    # length check) can catch it.
    with pytest.raises(ValueError):
        sfu.string_to_symplectic(["ZQ", "IZ"])


def test_string_to_symplectic_rejects_sign_character():
    with pytest.raises(ValueError):
        sfu.string_to_symplectic(["-ZI", "-IZ"])


def test_string_to_symplectic_rejects_trailing_newline():
    with pytest.raises(ValueError):
        sfu.string_to_symplectic(["ZI\n", "IZ\n"])


def test_string_to_symplectic_rejects_embedded_space():
    with pytest.raises(ValueError):
        sfu.string_to_symplectic(["Z I", "I Z"])


def test_string_to_symplectic_rejects_ragged_lengths():
    with pytest.raises(ValueError):
        sfu.string_to_symplectic(["ZI", "IZZ"])


def test_string_to_symplectic_ragged_and_bad_character_raises_value_error_with_message():
    # Both ragged and bad-charactered; must raise the new ValueError (naming the label
    # and/or the offending character), not an IndexError from the old truncating parser.
    with pytest.raises(ValueError) as exc_info:
        sfu.string_to_symplectic(["-ZI", "IZ"])

    message = str(exc_info.value)
    assert "-ZI" in message


def test_string_to_symplectic_empty_list_is_rejected():
    # An empty list means "no generators over an unknown number of qubits", which is
    # not something a tableau can be built from -- the width is exactly what is missing.
    # Returning zero-by-zero matrices instead would assert "over zero qubits", which the
    # function has no way to know, and would push the failure into a bare shape assert
    # further downstream.
    with pytest.raises(ValueError, match=r"at least one generator"):
        sfu.string_to_symplectic([])


def test_string_to_symplectic_rejects_bare_string():
    with pytest.raises(ValueError):
        sfu.string_to_symplectic("ZI")


def test_string_to_symplectic_round_trips_with_symplectic_to_string():
    rng = np.random.default_rng(0)
    for _ in range(20):
        n_row, n_col = rng.integers(1, 5), rng.integers(1, 5)
        x_matrix = rng.integers(0, 2, size=(n_row, n_col))
        z_matrix = rng.integers(0, 2, size=(n_row, n_col))
        labels = sfu.symplectic_to_string(x_matrix, z_matrix)

        round_trip_x, round_trip_z = sfu.string_to_symplectic(labels)

        assert np.array_equal(round_trip_x, x_matrix)
        assert np.array_equal(round_trip_z, z_matrix)


def test_from_labels_rejects_non_pauli_character():
    tableau = StabilizerTableau(2)

    with pytest.raises(ValueError):
        tableau.from_labels(["ZQ", "IZ"])


def test_get_stabilizer_element_by_string_rejects_sign_character():
    with pytest.raises(ValueError):
        sfu.get_stabilizer_element_by_string("-ZI")


def test_get_stabilizer_element_by_string_accepts_clean_input():
    element = sfu.get_stabilizer_element_by_string("ZI")

    assert element.shape == (4, 4)


def test_stabilizer_to_density_rejects_sign_character_labels():
    # A sign adds a column, which the square-table check would also reject; match the
    # character check's own message so the test pins the new rejection.
    with pytest.raises(ValueError, match="unexpected character"):
        rc.stabilizer_to_density(["-ZI", "-IZ"])


def test_stabilizer_to_density_rejects_stray_character_labels():
    with pytest.raises(ValueError):
        rc.stabilizer_to_density(["ZQ", "IZ"])
