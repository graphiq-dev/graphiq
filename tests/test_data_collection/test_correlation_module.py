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

from graphiq.backends.stabilizer.compiler import StabilizerCompiler
from graphiq.backends.stabilizer.functions.rep_conversion import (
    get_clifford_tableau_from_graph,
)
from graphiq.data_collection.correlation_module import graph_to_circ
from graphiq.metrics import Infidelity
from graphiq.state import QuantumState

_p3_plus_k1 = nx.disjoint_union(nx.path_graph(3), nx.empty_graph(1))


@pytest.mark.parametrize(
    "graph",
    [
        _p3_plus_k1,
        nx.disjoint_union(nx.empty_graph(1), nx.path_graph(3)),
        nx.to_numpy_array(_p3_plus_k1),
    ],
    ids=["isolate_last", "isolate_first", "adjacency_matrix"],
)
def test_graph_to_circ_rejects_a_graph_with_an_isolated_vertex(graph):
    with pytest.raises(ValueError, match="isolated"):
        graph_to_circ(graph)


@pytest.mark.parametrize("outcome", [0, 1])
def test_graph_to_circ_solves_a_disconnected_graph_without_an_isolated_vertex(outcome):
    graph = nx.disjoint_union(nx.path_graph(2), nx.path_graph(2))
    circuit = graph_to_circ(graph)

    target = QuantumState(get_clifford_tableau_from_graph(graph), rep_type="s")
    # force every measurement to 0, then to 1, so each outcome is checked every run
    compiler = StabilizerCompiler()
    compiler.measurement_determinism = outcome
    state = compiler.compile(circuit)
    n_photon = graph.number_of_nodes()
    state.partial_trace(list(range(n_photon)), [2] * circuit.n_quantum)
    assert np.isclose(Infidelity(target).evaluate(state, circuit), 0.0)
