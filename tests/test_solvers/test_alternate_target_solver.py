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
import pytest as pytest

from graphiq.backends.stabilizer.clifford_tableau import CliffordTableau
from graphiq.backends.stabilizer.functions.transformation import run_circuit
from graphiq.benchmarks.graph_states import linear_cluster_state
from graphiq.solvers.alternate_target_solver import *
from graphiq.utils.relabel_module import (
    _label_finder,
    _partial_orbit,
    automorph_check,
)

linear_cluster = nx.from_numpy_array(nx.to_numpy_array(linear_cluster_state(4).data))


# setting


def my_setting():
    solver_setting = AlternateTargetSolverSetting()
    # solver_setting options
    solver_setting.allow_relabel = True
    solver_setting.n_iso_graphs = 1
    solver_setting.rel_inc_thresh = 0.2
    solver_setting.allow_exhaustive = True
    solver_setting.iso_thresh = None
    solver_setting.n_lc_graphs = 1
    solver_setting.lc_orbit_depth = None
    solver_setting.lc_method = None
    solver_setting.depolarizing_rate = 0.005
    solver_setting.verbose = False
    solver_setting.save_openqasm = ""
    return solver_setting


# solver


# without noise
@pytest.mark.parametrize("target_graph", [linear_cluster, repeater_graph_states(3)])
def test_solver_no_noise(target_graph, setting=my_setting()):
    solver = AlternateTargetSolver(
        target=target_graph, solver_setting=setting, noise_model_mapping="depolarizing"
    )
    # solver options
    solver.metric = Infidelity
    solver.compiler = StabilizerCompiler()
    solver.noise_compiler = DensityMatrixCompiler()

    solver.io = None
    solver.noise_simulation = False
    solver.seed = 1
    solver.solve()


# with noise: Monte Carlo
@pytest.mark.parametrize("target_graph", [linear_cluster, repeater_graph_states(3)])
def test_solver_monte_carlo(target_graph, setting=my_setting()):
    setting.monte_carlo_params = {
        "n_sample": 20,
        "map": mcn.McNoiseMap(),
        "compiler": StabilizerCompiler(),
        "seed": 99,
        "n_parallel": 2,
        "n_single": 10,
    }
    setting.monte_carlo = True
    solver = AlternateTargetSolver(
        target=target_graph, solver_setting=setting, noise_model_mapping="depolarizing"
    )
    # solver options
    solver.metric = Infidelity
    solver.compiler = StabilizerCompiler()
    solver.noise_compiler = DensityMatrixCompiler()

    solver.io = None
    solver.noise_simulation = True
    solver.seed = 1
    solver.solve()


# with noise: density matrix noise
def test_solver_density_noise(setting=my_setting(), target_graph=linear_cluster):
    setting.monte_carlo = False
    solver = AlternateTargetSolver(
        target=target_graph, solver_setting=setting, noise_model_mapping="depolarizing"
    )
    # solver options
    solver.metric = Infidelity
    solver.compiler = StabilizerCompiler()
    solver.noise_compiler = DensityMatrixCompiler()

    solver.io = None
    solver.noise_simulation = True
    solver.seed = 1
    solver.solve()


def _solve_with_label_map(label_map, n_iso_graphs):
    setting = my_setting()
    setting.n_iso_graphs = n_iso_graphs
    setting.label_map = label_map
    solver = AlternateTargetSolver(
        target=linear_cluster,
        solver_setting=setting,
        noise_model_mapping="depolarizing",
    )
    solver.metric = Infidelity
    solver.compiler = StabilizerCompiler()
    solver.noise_compiler = DensityMatrixCompiler()
    solver.io = None
    solver.noise_simulation = False
    solver.seed = 1
    return solver.solve()


@pytest.mark.parametrize("n_iso_graphs", [3, 12])
def test_label_map_setting_gives_the_same_result_as_without_it(n_iso_graphs):
    # With label_map set, iso_finder returns (matrices, label maps). With seed 1, 3
    # graphs are found in its first batch of labellings and 12 need a longer search,
    # so both of its return paths are covered.
    setting = my_setting()
    adj = nx.to_numpy_array(linear_cluster)
    first_batch = automorph_check(adj, _label_finder(n_iso_graphs, len(adj), seed=1))
    assert (len(first_batch) >= n_iso_graphs) == (n_iso_graphs == 3)
    found = iso_finder(
        adj,
        n_iso_graphs,
        rel_inc_thresh=setting.rel_inc_thresh,
        allow_exhaustive=setting.allow_exhaustive,
        sort_emit=setting.sort_emitter,
        label_map=True,
        thresh=setting.iso_thresh,
        seed=1,
    )
    assert isinstance(found, tuple)
    assert len(found[1]) == len(found[0])

    with_map = _solve_with_label_map(True, n_iso_graphs)
    without_map = _solve_with_label_map(False, n_iso_graphs)

    assert len(with_map) == len(without_map) == n_iso_graphs
    for (circuit_a, props_a), (circuit_b, props_b) in zip(with_map, without_map):
        assert nx.utils.graphs_equal(props_a["g"], props_b["g"])
        assert props_a["map"] == props_b["map"]
        assert props_a["score"] == props_b["score"]
        assert [type(op) for op in circuit_a.sequence()] == [
            type(op) for op in circuit_b.sequence()
        ]


def test_setting_callback_func_is_per_instance():
    a = AlternateTargetSolverSetting()
    b = AlternateTargetSolverSetting()
    assert a.callback_func is not b.callback_func
    assert a.callback_func == {}

    a.callback_func["x"] = 1
    assert b.callback_func == {}

    d = {}
    setting = AlternateTargetSolverSetting(callback_func=d)
    assert setting.callback_func is d


def test_default_setting_names_a_dispatchable_lc_method():
    # every other test sets lc_method by hand, so an undispatchable default
    # went unnoticed
    setting = AlternateTargetSolverSetting()
    assert setting.lc_method is None

    solver = AlternateTargetSolver(
        target=nx.path_graph(4), solver_setting=setting, seed=1
    )
    solver.solve()
    assert solver.result is not None


def test_allow_relabel_false_searches_only_the_given_labelling():
    setting = AlternateTargetSolverSetting(n_iso_graphs=3, n_lc_graphs=2)
    setting.allow_relabel = False
    setting.lc_method = None
    solver = AlternateTargetSolver(
        target=nx.path_graph(4), solver_setting=setting, seed=1
    )
    results = solver.solve()
    n = solver.target_graph.number_of_nodes()
    for _, properties in results:
        # get_relabel_map also carries a -1 "self" key, so identity is over range(n)
        assert {i: properties["map"][i] for i in range(n)} == {i: i for i in range(n)}


def test_linear_lc_method_on_a_path_not_in_iteration_order():
    # nodes 0..5 inserted in order, path 2-0-4-1-5-3
    order = [2, 0, 4, 1, 5, 3]
    target = nx.Graph()
    target.add_nodes_from(range(6))
    target.add_edges_from(zip(order, order[1:]))
    setting = AlternateTargetSolverSetting(n_iso_graphs=1, n_lc_graphs=10)
    setting.allow_relabel = False
    setting.lc_method = "linear"
    solver = AlternateTargetSolver(target=target, solver_setting=setting, seed=1)
    results = solver.solve()
    candidates = {frozenset(map(frozenset, r[1]["g"].edges())) for r in results}
    assert len(candidates) == len(_partial_orbit(6))


def test_allow_lc_false_keeps_the_isomorphic_graph_unchanged():
    setting = AlternateTargetSolverSetting(n_iso_graphs=1, n_lc_graphs=4)
    setting.allow_relabel = False
    setting.allow_lc = False
    solver = AlternateTargetSolver(
        target=nx.path_graph(4), solver_setting=setting, seed=1
    )
    results = solver.solve()
    assert len(results) == 1
    # the candidate is rebuilt from an adjacency matrix, so it carries weight
    # attributes the target does not; compare structure, not edge data
    candidate = results[0][1]["g"]
    assert sorted(candidate.nodes()) == sorted(solver.target_graph.nodes())
    assert sorted(map(sorted, candidate.edges())) == sorted(
        map(sorted, solver.target_graph.edges())
    )


def test_allow_lc_false_bypasses_lc_method_validation():
    setting = AlternateTargetSolverSetting(n_iso_graphs=1, n_lc_graphs=1)
    setting.allow_relabel = False
    setting.allow_lc = False
    setting.lc_method = "no such search"
    solver = AlternateTargetSolver(
        target=nx.path_graph(4), solver_setting=setting, seed=1
    )
    solver.solve()
    assert solver.result is not None


def test_mc_depol_builds_the_expected_noise_map():
    # solver_setting.monte_carlo = True, set before construction, is what makes __init__ set
    # depolarizing_rate and build the Monte Carlo noise map via mc_depol() (rather than the
    # density-matrix noise_model_mapping the other tests here use).
    setting = my_setting()
    setting.monte_carlo = True
    solver = AlternateTargetSolver(
        target=linear_cluster,
        solver_setting=setting,
        noise_model_mapping="depolarizing",
    )

    mapping = solver.mc_params["map"].mapping
    assert set(mapping.keys()) == {"e", "p", "ee", "ep"}
    assert mapping["p"] == {}
    assert set(mapping["e"].keys()) == {
        "Hadamard",
        "Phase",
        "PhaseDagger",
        "SigmaX",
        "SigmaY",
        "SigmaZ",
    }
    assert set(mapping["ee"].keys()) == {"CNOT"}
    assert set(mapping["ep"].keys()) == {"CNOT"}

    rate = solver.depolarizing_rate
    assert rate == setting.depolarizing_rate
    per_gate_rate = rate / 3
    gate_entries = [("e", name) for name in mapping["e"]] + [
        ("ee", "CNOT"),
        ("ep", "CNOT"),
    ]
    for reg_type, gate_name in gate_entries:
        entries = mapping[reg_type][gate_name]
        assert len(entries) == 3
        assert {noise.noise_parameters["Pauli error"] for noise, _ in entries} == {
            "X",
            "Y",
            "Z",
        }
        for _, prob in entries:
            assert prob == pytest.approx(per_gate_rate)

    tuples = solver.mc_params["map"].get_gate_noise("e", "Hadamard")
    assert len(tuples) == 4
    pauli_tuples, (no_noise, no_noise_prob) = tuples[:3], tuples[3]
    assert {t[0].noise_parameters["Pauli error"] for t in pauli_tuples} == {
        "X",
        "Y",
        "Z",
    }
    for _, prob in pauli_tuples:
        assert prob == pytest.approx(per_gate_rate)
    assert isinstance(no_noise, nm.NoNoise)
    # The residual is built as 1 - 3 * (rate / 3), which differs from 1 - rate by one ulp
    # for some rates, so the comparison is approximate rather than bit-exact.
    assert no_noise_prob == pytest.approx(1 - rate)


# graph_to_circ input checks

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


def test_a_stabilizer_target_not_in_standard_form_is_accepted():
    # the star graph state (centre 0) as a circuit leaves it: generators XZZ, ZXI, IXX
    tableau = run_circuit(
        CliffordTableau(3),
        [("H", 0), ("CNOT", 0, 1), ("CNOT", 1, 2), ("H", 1), ("H", 2)],
    )

    solver = AlternateTargetSolver(target=QuantumState(tableau, rep_type="s"))

    assert sorted(solver.target_graph.edges) == [(0, 1), (0, 2)]
