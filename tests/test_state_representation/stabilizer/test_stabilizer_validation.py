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
Tests for sfu.is_stabilizer and StabilizerTableau.validate.
"""

import numpy as np
import pytest

import graphiq.backends.stabilizer.functions.utils as sfu
from graphiq.backends.stabilizer.tableau import StabilizerTableau


@pytest.mark.parametrize(
    "labels,expected",
    [
        (["ZII", "IZI", "IIZ"], True),
        (["XXX", "ZZI", "IZZ"], True),  # GHZ
        (["ZII", "ZII", "IIZ"], False),  # duplicate row -> not independent
        (["XII", "ZII", "IIZ"], False),  # X and Z on qubit 0 anticommute
        (["ZZI", "IZZ", "ZIZ"], False),  # commuting, GF(2) rank 2, real rank 3
        (["III", "IZI", "IIZ"], False),  # zero row
    ],
)
def test_is_stabilizer(labels, expected):
    x_matrix, z_matrix = sfu.string_to_symplectic(labels)
    table = np.block([x_matrix, z_matrix]).astype(int)
    assert sfu.is_stabilizer(table) is expected


def test_stabilizer_tableau_validate():
    tableau = StabilizerTableau(3)
    tableau.from_labels(["ZII", "IZI", "IIZ"])
    assert tableau.validate() is True


def test_stabilizer_tableau_validate_false_for_dependent_rows():
    tableau = StabilizerTableau(3)
    tableau.from_labels(["ZII", "ZII", "IIZ"])
    assert tableau.validate() is False

