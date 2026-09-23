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

from graphiq.benchmarks.graph_states import linear_cluster_state
from graphiq.solvers.alternate_target_solver import *

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
