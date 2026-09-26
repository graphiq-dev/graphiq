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
import math

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pytest

from graphiq.backends.stabilizer.functions.height import height_max
from graphiq.benchmarks.graph_states import linear_cluster_state
from graphiq.utils.relabel_module import *
from graphiq.utils.relabel_module import _compare_graphs_visual, _label_finder


def random_connected_graph(n, p, seed=None):
    rnd_maker = nx.gnp_random_graph
    G = nx.Graph()
    G.add_edges_from(((0, 1), (2, 3)))
    while not nx.is_connected(G):
        G = rnd_maker(n, p, seed=seed, directed=False)
    adj_G = nx.to_numpy_array(G)
    return adj_G


@pytest.mark.parametrize("n_node", [5, 6, 7, 8])
def test_random_graph(n_node):
    n_iso = math.factorial(n_node - 2)
    for p in range(4, 7):
        found = []
        for i in range(3):
            adj = random_connected_graph(n_node, p / 10)
            iso_found = len(iso_finder(adj, n_iso)) / n_iso
            found.append(iso_found)
        avg = sum(found) / len(found)
        assert 0 <= avg <= 1


@pytest.mark.parametrize("n_node", [11, 12, 13, 15])
@pytest.mark.parametrize("sort_emit", [True, False])
def test_relabel(n_node, sort_emit):
    # find 100 relabeled graphs for an initial random one with n_nodes vertices.
    adj = random_connected_graph(n_node, 0.5)
    n_iso = 100
    adj_arr = iso_finder(adj, n_iso, sort_emit=sort_emit)
    assert 1 <= len(adj_arr) <= 100
    assert adj_arr[10].shape[1] == n_node or len(adj_arr) == 0


def test_complete_graph():
    # there should be only 1 valid result for relabeling of a complete graph
    for n_node in range(5, 15):
        g = nx.complete_graph(n_node)
        adj = nx.to_numpy_array(g)
        n_iso = 10
        adj_arr = iso_finder(adj, n_iso)
        assert len(adj_arr) == 1


def test_emitter_sort():
    for n_node in range(4, 8):
        adj = random_connected_graph(n_node, 0.5)
        adj_arr = iso_finder(adj, 20)
        sorted_list = emitter_sorted(adj_arr)
        n_emit_list = [x[1] for x in sorted_list]
        if sorted_list:
            assert all(
                n_emit_list[i] <= n_emit_list[i + 1]
                for i in range(len(n_emit_list) - 1)
            )
            assert isinstance(sorted_list[0][0], np.ndarray)


def _linear_cluster_adj(n):
    return nx.to_numpy_array(linear_cluster_state(n).data)


def _returns_on_first_batch(adj, n_iso, seed):
    # iso_finder returns straight after its first batch of labellings when that
    # batch already holds n_iso distinct graphs
    labels = _label_finder(n_iso, adj.shape[0], seed=seed)
    return len(automorph_check(adj, labels)) >= n_iso


def _emitter_counts(adj_arr):
    return [height_max(graph=nx.from_numpy_array(adj)) for adj in adj_arr]


def _assert_maps_pair_with_matrices(adj, found):
    assert isinstance(found, tuple)
    adj_arr, maps = found
    assert len(maps) == len(adj_arr)
    g = nx.from_numpy_array(adj)
    for new_adj, mapping in zip(adj_arr, maps):
        new_g = nx.from_numpy_array(new_adj)
        mapped_edges = {frozenset((mapping[u], mapping[v])) for u, v in g.edges}
        assert mapped_edges == {frozenset(e) for e in new_g.edges}


def test_sort_emit_and_label_map_apply_when_the_first_batch_is_enough():
    adj = _linear_cluster_adj(6)
    kwargs = dict(rel_inc_thresh=0.1, allow_exhaustive=False, sort_emit=True, seed=1)
    assert _returns_on_first_batch(adj, 10, seed=1)

    adj_arr = iso_finder(adj, 10, **kwargs)
    n_emit = _emitter_counts(adj_arr)
    assert n_emit == sorted(n_emit)
    assert n_emit != _emitter_counts(
        iso_finder(adj, 10, **{**kwargs, "sort_emit": False})
    )

    # the maps must pair with the sorted matrices, not the order they were found in
    _assert_maps_pair_with_matrices(adj, iso_finder(adj, 10, label_map=True, **kwargs))


def test_label_map_applies_when_the_first_batch_is_enough():
    adj = _linear_cluster_adj(4)
    assert _returns_on_first_batch(adj, 3, seed=1)

    found = iso_finder(adj, 3, label_map=True, seed=1)

    _assert_maps_pair_with_matrices(adj, found)
    assert len(found[0]) == 3


def test_label_map_has_one_map_per_returned_matrix_after_a_longer_search():
    adj = _linear_cluster_adj(4)
    assert not _returns_on_first_batch(adj, 5, seed=5)

    found = iso_finder(adj, 5, sort_emit=False, label_map=True, seed=5)

    _assert_maps_pair_with_matrices(adj, found)
    assert len(found[0]) == 5


def test_sort_emit_and_label_map_apply_at_the_high_symmetry_stop():
    adj = nx.to_numpy_array(nx.cycle_graph(4))
    with pytest.warns(UserWarning, match="high symmetry"):
        found = iso_finder(
            adj, 5, allow_exhaustive=False, sort_emit=True, label_map=True, seed=1
        )

    _assert_maps_pair_with_matrices(adj, found)
    n_emit = _emitter_counts(found[0])
    assert n_emit == sorted(n_emit)
    assert len(set(n_emit)) > 1  # mixed counts, so the sort is actually exercised


def test_compare_graphs_visual_draws_the_three_panels(monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    g = nx.path_graph(4)
    new_labels = [2, 0, 3, 1]
    new_g = nx.relabel_nodes(g, dict(enumerate(new_labels)))
    plt.close("all")
    try:
        _compare_graphs_visual(g, new_g, new_labels)
        panels = [
            plt.figure(n) for n in plt.get_fignums() if len(plt.figure(n).axes) == 3
        ]
        assert len(panels) == 1
        assert all(ax.collections for ax in panels[0].axes)
    finally:
        plt.close("all")


def test_compare_graphs_visual_rejects_repeated_labels(monkeypatch):
    # relabel_nodes merges nodes that share a label, so the relabeled graph
    # does not have exactly new_labels as its nodes
    monkeypatch.setattr(plt, "show", lambda: None)
    g = nx.path_graph(4)
    new_labels = [0, 0, 1, 2]
    new_g = nx.relabel_nodes(g, dict(enumerate(new_labels)))
    try:
        with pytest.raises(AssertionError):
            _compare_graphs_visual(g, new_g, new_labels)
    finally:
        plt.close("all")
