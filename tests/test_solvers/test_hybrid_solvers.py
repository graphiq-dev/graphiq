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
import numpy as np
import networkx as nx
import pytest
from graphiq.benchmarks.circuits import (
    linear_cluster_4qubit_circuit,
    linear_cluster_3qubit_circuit,
    bell_state_circuit,
    ghz3_state_circuit,
)
from graphiq.state import QuantumState
from graphiq.metrics import Infidelity
from graphiq.backends.density_matrix.compiler import DensityMatrixCompiler
from graphiq.backends.stabilizer.compiler import StabilizerCompiler
from graphiq.backends.state_rep_conversion import graph_to_density
from graphiq.backends.stabilizer.functions.rep_conversion import (
    get_clifford_tableau_from_graph,
)
from graphiq import noise
from graphiq.solvers.evolutionary_solver import (
    EvolutionarySolver,
    EvolutionarySolverSetting,
)
from graphiq.solvers.hybrid_solvers import HybridEvolutionarySolver
from graphiq.benchmarks.graph_states import repeater_graph_states
from graphiq.benchmarks.alternate_circuits import (
    noise_model_loss_and_depolarizing,
    noise_model_pure_loss,
    exemplary_test,
)


def graph_stabilizer_setup(graph, solver_class, solver_setting, expected_result):
    """
    A common piece of code to run a graph state with the stabilizer backend

    :param graph: the graph rep_type of a graph state
    :type graph: networkX.Graph
    :param solver_class: a solver class to run
    :type solver_class: a subclass of SolverBase
    :param solver_setting:
    :type solver_setting:
    :param expected_result:
    :type expected_result:
    :return: the solver instance
    :rtype: SolverBase
    """
    target_tableau = get_clifford_tableau_from_graph(graph)

    target = QuantumState(target_tableau, rep_type="s")
    compiler = StabilizerCompiler()
    compiler.measurement_determinism = 1
    metric = Infidelity(target)

    for random_seed in range(0, 500, 10):
        solver = solver_class(
            target=target,
            metric=metric,
            compiler=compiler,
            solver_setting=solver_setting,
        )
        solver.seed(random_seed)
        solver.solve()
        score, circuit = solver.result

        if score == expected_result:
            return solver
    return solver


def noise_model_loss_and_depolarizing_2(error_rate, loss_rate):
    emitter_noise = noise.DepolarizingNoise(error_rate)
    photon_loss = noise.PhotonLoss(loss_rate)
    noise_model_mapping = {
        "e": {},
        "p": {"SigmaX": photon_loss, "SigmaY": photon_loss, "SigmaZ": photon_loss},
        "ee": {},
        "ep": {},
    }
    return noise_model_mapping


def test_repeater_graph_state_4():
    graph = repeater_graph_states(4)
    solver_setting = EvolutionarySolverSetting()
    solver = graph_stabilizer_setup(
        graph, HybridEvolutionarySolver, solver_setting, 0.0
    )
    score, circuit = solver.result
    assert np.allclose(score, 0.0)
    # circuit.draw_circuit()


def test_square4():
    graph = nx.Graph([(1, 2), (2, 3), (2, 4), (4, 3), (1, 3)])
    solver_setting = EvolutionarySolverSetting()
    solver = graph_stabilizer_setup(
        graph, HybridEvolutionarySolver, solver_setting, 0.0
    )
    score, circuit = solver.result
    assert np.allclose(score, 0.0)
    # circuit.draw_circuit()


def test_alternate_circuits_1():
    graph = repeater_graph_states(3)
    solver_setting = EvolutionarySolverSetting()
    solver_setting.n_hof = 10
    solver_setting.n_pop = 60
    solver_setting.n_stop = 60
    noise_model = noise_model_loss_and_depolarizing(0, 0)
    exemplary_test(graph, noise_model, solver_setting, 1000)


def test_alternate_circuits_2():
    graph = repeater_graph_states(4)
    noise_model = noise_model_loss_and_depolarizing(0, 0)
    exemplary_test(graph, noise_model, None, 1000)


def test_alternate_circuits_w_noise():
    loss_rate = 0.01
    graph = repeater_graph_states(4)
    noise_model = noise_model_pure_loss(loss_rate)
    exemplary_test(graph, noise_model, None, 1000)


def test_alternate_circuits_w_noise2():
    error_rate = 0.01
    loss_rate = 0.01
    graph = repeater_graph_states(3)
    noise_model = noise_model_loss_and_depolarizing(error_rate, loss_rate)
    # The depolarizing noise on every emitter CNOT leg folds each noisy compile into a
    # large mixed stabilizer, so the default 50x50 search is very slow. This is a smoke
    # test that the noisy hybrid search completes, not a search-quality test, so it runs
    # 50 candidates instead of 2,500.
    solver_setting = EvolutionarySolverSetting()
    solver_setting.n_pop = 10
    solver_setting.n_stop = 5
    exemplary_test(graph, noise_model, solver_setting, 1)


def test_alternate_circuits_w_noise3():
    error_rate = 0.01
    loss_rate = 0.01
    graph = nx.star_graph(3)
    noise_model = noise_model_loss_and_depolarizing(error_rate, loss_rate)
    exemplary_test(graph, noise_model, None, 1000)


def _solve_path3(target, compiler):
    metric = Infidelity(target)
    solver = HybridEvolutionarySolver(
        target=target,
        metric=metric,
        compiler=compiler,
        solver_setting=EvolutionarySolverSetting(n_stop=3, n_pop=5),
    )
    HybridEvolutionarySolver.seed(3)
    solver.solve()
    return solver, metric


def test_a_density_matrix_target_with_the_density_matrix_compiler_is_solved():
    # the deterministic solver seeding the population converts its target to "s" in
    # place; the metric sharing this target must keep scoring density matrices
    target = QuantumState(graph_to_density(nx.path_graph(3)), rep_type="dm")

    solver, metric = _solve_path3(target, DensityMatrixCompiler())

    assert target.rep_type == "dm"
    assert metric.target is target
    assert all(0 <= score <= 1 for score, _ in solver.hof)
    # the first evaluation is the deterministic circuit, which prepares the target
    assert metric.log[0] == pytest.approx(0, abs=1e-9)


def test_a_density_matrix_target_with_the_stabilizer_compiler_is_unchanged():
    target = QuantumState(graph_to_density(nx.path_graph(3)), rep_type="dm")

    solver, _ = _solve_path3(target, StabilizerCompiler())

    assert [score for score, _ in solver.hof] == pytest.approx(
        [0.5, 0.5, 0.75, 0.75, 0.75]
    )
    # the stabilizer compiler's states are still scored against the converted target;
    # this assertion, not the scores, is what tells the two metric arms apart
    assert target.rep_type == "s"


def test_a_graph_target_with_the_stabilizer_compiler_still_solves():
    target = QuantumState(nx.path_graph(3), rep_type="g")

    solver, _ = _solve_path3(target, StabilizerCompiler())

    assert [score for score, _ in solver.hof] == pytest.approx(
        [0.5, 0.5, 0.75, 0.75, 0.75]
    )


def _path3_hybrid(noise_model_mapping=None):
    target = QuantumState(
        get_clifford_tableau_from_graph(nx.path_graph(3)), rep_type="s"
    )
    compiler = StabilizerCompiler()
    solver = HybridEvolutionarySolver(
        target=target,
        metric=Infidelity(target),
        compiler=compiler,
        noise_model_mapping=noise_model_mapping,
        solver_setting=EvolutionarySolverSetting(n_stop=2, n_pop=2),
    )
    return solver, compiler


def _record_search_compiles(solver, compiler, monkeypatch):
    # record the `mixed` flag of every state compiled after the population is
    # initialised; the deterministic solve that seeds it compiles first
    recorded = []
    initialize = solver.population_initialization
    compile_ = compiler.compile

    def recording_compile(circuit, initial_state=None):
        state = compile_(circuit, initial_state)
        recorded.append(state.mixed)
        return state

    def initialize_then_record():
        population = initialize()
        monkeypatch.setattr(compiler, "compile", recording_compile)
        return population

    monkeypatch.setattr(solver, "population_initialization", initialize_then_record)
    return recorded


def test_hybrid_without_noise_leaves_the_compiler_noise_free():
    solver, compiler = _path3_hybrid()
    HybridEvolutionarySolver.seed(0)
    solver.solve()
    assert solver.noise_simulation is False
    assert compiler.noise_simulation is False
    assert compiler.compile(solver.result[1]).mixed is False


def test_hybrid_without_noise_searches_with_noise_free_compiles(monkeypatch):
    solver, compiler = _path3_hybrid()
    recorded = _record_search_compiles(solver, compiler, monkeypatch)
    HybridEvolutionarySolver.seed(0)
    solver.solve()
    assert recorded
    assert not any(recorded)


def test_hybrid_with_a_noise_mapping_still_simulates_noise(monkeypatch):
    solver, compiler = _path3_hybrid(
        noise_model_mapping={"e": {}, "p": {}, "ee": {}, "ep": {}}
    )
    recorded = _record_search_compiles(solver, compiler, monkeypatch)
    HybridEvolutionarySolver.seed(0)
    solver.solve()
    assert solver.noise_simulation is True
    assert recorded
    assert all(recorded)
