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
from graphiq.backends.graph.state import Graph, MixedGraph
import networkx as nx


def test_graph_initialization():
    graph = nx.Graph([(0, 1), (1, 2), (1, 3), (2, 3)])
    state = Graph(graph)
    state.draw()
    assert state.n_qubits == 4
    print(f"Neighbors of node 1 are {state.get_neighbors(1)} ")
    lc_state = state.local_complementation(1, copy=True)
    state.draw()
    lc_state.draw()


def test_graph_lc():
    lc_dict = {0: ["x"], 1: ["h"], 2: ["s"], 3: ["s", "h"]}
    graph = nx.Graph([(0, 1), (1, 2), (1, 3), (2, 3)])
    state = Graph(graph, lc_dict)
    print(f"Local Clifford gate at node 1 is {state.find_lc(1)}")


def test_mixed_graph():
    graph1 = nx.Graph([(0, 1), (1, 2), (1, 3), (2, 3)])
    state1 = Graph(graph1)
    graph2 = nx.Graph([(0, 2), (1, 2), (0, 3), (2, 3)])
    state2 = Graph(graph2)
    state3 = MixedGraph([(0.5, state1), (0.5, state2)])
    print(state3.mixture)
    assert state3.n_qubits == 4
    assert state3.probability == 1.0