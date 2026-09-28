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
``state_rep_conversion.stabilizer_to_density`` reads the phase vector.

Every expected density matrix below is written out by hand from the state it
describes.
"""

import networkx as nx
import numpy as np
import pytest

import graphiq.backends.state_rep_conversion as rc
from graphiq.backends.stabilizer.clifford_tableau import CliffordTableau
from graphiq.backends.stabilizer.tableau import StabilizerTableau
from graphiq.state import QuantumState


def test_stabilizer_to_density_converts_ket_one_correctly():
    # ['-Z'], phase [1]: the state |1>. Sign-blind, this reads as ['+Z'] = |0>.
    tableau = StabilizerTableau([np.array([[0]]), np.array([[1]])], np.array([1]))

    rho = rc.stabilizer_to_density(tableau)

    expected = np.array([[0, 0], [0, 1]], dtype=complex)
    assert np.allclose(rho, expected)


def test_stabilizer_to_density_handles_a_two_qubit_signed_generator_set():
    # ['ZZ', '-XX']: (|00> - |11>) / sqrt(2). Sign-blind, the second generator's
    # sign is dropped and this reads as (|00> + |11>) / sqrt(2) instead.
    x_matrix = np.array([[0, 0], [1, 1]])
    z_matrix = np.array([[1, 1], [0, 0]])
    phase = np.array([0, 1])
    tableau = StabilizerTableau([x_matrix, z_matrix], phase)

    rho = rc.stabilizer_to_density(tableau)

    expected = np.zeros((4, 4), dtype=complex)
    expected[0, 0] = 0.5
    expected[3, 3] = 0.5
    expected[0, 3] = -0.5
    expected[3, 0] = -0.5
    assert np.allclose(rho, expected)


def test_stabilizer_to_density_leaves_a_graph_state_unchanged():
    # Graph states carry an all-zero phase vector, so this is the case the
    # fix must not disturb.
    graph = nx.Graph([(0, 1), (1, 2), (2, 0)])
    tableau = rc.graph_to_stabilizer(graph)[0][1]

    rho = rc.stabilizer_to_density(tableau)

    assert np.allclose(rho, rc.graph_to_density(graph))


def test_conversion_rejects_a_nonzero_i_phase():
    # StabilizerTableau has no iphase -- CliffordTableau.to_stabilizer() drops
    # it -- so the check has to happen on the CliffordTableau. A generator
    # carrying a factor of i cannot be read as a bare +1/-1 sign.
    tableau = CliffordTableau(1)
    tableau.iphase[tableau.n_qubits] = 1

    state = QuantumState(tableau, rep_type="s")

    with pytest.raises(ValueError, match="i phase"):
        state.convert_representation("dm")


def test_conversion_rejects_a_nonzero_destabilizer_i_phase():
    tableau = CliffordTableau(1)
    tableau.iphase[0] = 1

    state = QuantumState(tableau, rep_type="s")

    with pytest.raises(ValueError, match="i phase"):
        state.convert_representation("dm")


def test_conversion_rejects_a_nonzero_i_phase_in_a_mixture():
    good = CliffordTableau(1)
    bad = CliffordTableau(1)
    bad.iphase[bad.n_qubits] = 1

    state = QuantumState([(0.5, good), (0.5, bad)], rep_type="s", mixed=True)

    with pytest.raises(ValueError, match="i phase"):
        state.convert_representation("dm")
