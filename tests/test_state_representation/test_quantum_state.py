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
import networkx as nx
import numpy as np
import pytest

from graphiq.backends.density_matrix import numpy as dmnp
from graphiq.backends.stabilizer.clifford_tableau import CliffordTableau
from graphiq.state import QuantumState


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
def test_initializing_dm_1(n):
    data = dmnp.eye(2**n)
    state = QuantumState(data, rep_type="dm")
    assert np.allclose(state.rep_data.data, data / np.trace(data))


def test_initializing_dm_2():
    """Test that it fails if we don't provide the right datatype"""
    data = "bad input"
    with pytest.raises(TypeError):
        QuantumState(data, rep_type="dm")


def test_initializing_graph_2():
    n = 3
    data = nx.Graph([(1, 2), (2, 3)])
    state = QuantumState(data, rep_type="g")
    assert len(state.rep_data.data.nodes) == n


def test_initializing_stabilizer_1():
    """Checks that we can initialize stabilizers with integers"""
    n = 130
    state = QuantumState(n, rep_type="s")
    assert state.rep_data.n_qubits == n


def test_initializing_stabilizer_2():
    """Checks that we can initialize stabilizers with a CliffordTableau"""
    n = 130
    state = QuantumState(CliffordTableau(n), rep_type="s")
    assert state.rep_data.n_qubits == n
