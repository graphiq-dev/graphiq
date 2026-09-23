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

from graphiq.benchmarks import graph_states as gs


@pytest.mark.parametrize("n_list", [[2, 3], [2, 3, 2], [3, 3, 3], [1, 2, 1]])
def test_crazy(n_list):
    graph = gs.crazy(n_list)
    assert graph.number_of_nodes() == sum(n_list)

    # consecutive layers are complete bipartite and there are no intra-layer edges
    layers = []
    start = 0
    for n in n_list:
        layers.append(list(range(start, start + n)))
        start += n
    for layer in layers:
        for u in layer:
            for v in layer:
                if u != v:
                    assert not graph.has_edge(u, v)
    for layer_a, layer_b in zip(layers, layers[1:]):
        for u in layer_a:
            for v in layer_b:
                assert graph.has_edge(u, v)

    alternate_graph = gs.crazy(n_list, alternate_order=True)
    assert nx.is_isomorphic(graph, alternate_graph)
