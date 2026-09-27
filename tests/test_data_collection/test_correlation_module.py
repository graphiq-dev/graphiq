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

import graphiq.circuit.ops as ops
from graphiq.backends.stabilizer.compiler import StabilizerCompiler
from graphiq.backends.stabilizer.functions.rep_conversion import (
    get_clifford_tableau_from_graph,
)
from graphiq.circuit.circuit_dag import CircuitDAG
from graphiq.data_collection.correlation_module import (
    GraphCorr,
    NodeCorr,
    graph_to_circ,
    graph_to_cnot,
    graph_to_emit_depth,
    rnd_graph,
)
from graphiq.metrics import CircuitMaxEmitDepth, Infidelity
from graphiq.state import QuantumState


def test_graph_corr_initial_graph_is_none_without_an_initial_graph():
    assert GraphCorr(graph_list=[nx.path_graph(3)]).initial_graph is None
    assert GraphCorr().initial_graph is None


def test_num_isomorph_setter_without_an_initial_graph_sets_the_relabel_count(
    monkeypatch,
):
    graph = nx.cycle_graph(6)
    corr = GraphCorr(circ_metric="num_emit", graph_list=[graph], relabel_trials=1)
    corr.num_isomorph = 3
    assert corr.num_isomorph == 3
    assert corr.graph_list == [graph]

    counts = []

    def record(g, count):
        counts.append(count)
        return [g]

    monkeypatch.setattr(GraphCorr, "_graph_list_maker", staticmethod(record))
    corr._min_circ_met_over_relabel(graph)
    assert counts == [3]


def test_relabel_trials_work_after_graph_circ_dict(monkeypatch):
    graph = nx.cycle_graph(6)
    perm = [0, 1, 2, 3, 5, 4]
    adj = nx.to_numpy_array(graph)
    relabeled = nx.from_numpy_array(adj[np.ix_(perm, perm)])
    assert graph_to_cnot(graph) == 3
    assert graph_to_cnot(relabeled) == 2
    monkeypatch.setattr(
        GraphCorr, "_graph_list_maker", staticmethod(lambda g, count: [g, relabeled])
    )

    corr = GraphCorr(
        circ_metric="cnot", graph_list=[graph], relabel_trials=2, num_isomorph=1
    )
    corr.graph_circ_dict()
    assert relabeled not in corr._graph_circ_dictionary

    assert corr._min_circ_met_over_relabel(graph) == 2


def test_node_corr_graph_setter_updates_the_adjacency_matrix():
    corr = NodeCorr(nx.path_graph(4), relabel_trials=3, show_plot=False)
    new_graph = nx.cycle_graph(5)
    corr.graph = new_graph

    assert np.array_equal(corr.adj, nx.to_numpy_array(new_graph))
    assert corr.adj_list
    for adj in corr.adj_list:
        assert nx.is_isomorphic(nx.from_numpy_array(adj), new_graph)


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


def test_rnd_graph_with_a_seed_redraws_until_connected(monkeypatch):
    n, p, seed = 6, 0.3, 0
    assert not nx.is_connected(nx.gnp_random_graph(n, p, seed=seed))

    draw = nx.gnp_random_graph
    calls = []

    def bounded(*args, **kwargs):
        calls.append(1)
        if len(calls) > 100:
            raise RuntimeError("rnd_graph drew 100 graphs without a connected one")
        return draw(*args, **kwargs)

    monkeypatch.setattr(nx, "gnp_random_graph", bounded)
    graph = rnd_graph(n, p, seed=seed)
    assert nx.is_connected(graph)
    assert graph.number_of_nodes() == n
    assert sorted(rnd_graph(n, p, seed=seed).edges()) == sorted(graph.edges())


def test_rnd_graph_with_a_seed_keeps_a_connected_first_draw():
    n, p = 6, 0.3
    seeds = [s for s in range(50) if nx.is_connected(nx.gnp_random_graph(n, p, seed=s))]
    assert seeds
    for s in seeds:
        expected = nx.gnp_random_graph(n, p, seed=s)
        assert sorted(rnd_graph(n, p, seed=s).edges()) == sorted(expected.edges())


def test_cnot_per_photon_divides_the_cnot_count_by_the_number_of_photons():
    graph = nx.cycle_graph(6)
    n_cnot = graph_to_cnot(graph)
    assert n_cnot > 0
    corr = GraphCorr(circ_metric="cnot_per_photon", graph_list=[graph])
    assert corr._circ_met_value(graph) == pytest.approx(n_cnot / 6)


def test_graph_to_emit_depth_counts_only_the_gates_on_the_emitter():
    circuit = CircuitDAG(n_emitter=1, n_photon=0, n_classical=0)
    circuit.add(ops.Hadamard(register=0, reg_type="e"))
    circuit.add(ops.SigmaX(register=0, reg_type="e"))
    assert graph_to_emit_depth(circuit) == 2

    graph_circuit = graph_to_circ(nx.cycle_graph(6))
    assert graph_to_emit_depth(graph_circuit) == CircuitMaxEmitDepth().evaluate(
        None, graph_circuit
    )
