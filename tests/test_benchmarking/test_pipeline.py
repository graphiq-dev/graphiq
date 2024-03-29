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
import graphiq.noise.monte_carlo_noise as mcn
from graphiq.backends.stabilizer.compiler import StabilizerCompiler
from graphiq.backends.stabilizer.functions.rep_conversion import (
    get_clifford_tableau_from_graph,
)
from graphiq.io import IO
from graphiq.metrics import Infidelity
from graphiq.solvers.alternate_target_solver import (
    AlternateTargetSolverSetting,
    AlternateTargetSolver,
)
from graphiq.state import QuantumState
from graphiq.benchmarks.graph_states import linear_cluster_state
from graphiq.benchmarks.pipeline import benchmark_run_graph_search_solver


def test_benchmark_run_graph_search_solver():
    target_graph = linear_cluster_state(4)
    target_graph = target_graph.data
    target_tableau = get_clifford_tableau_from_graph(target_graph)
    target_state = QuantumState(target_tableau, rep_type="s")

    compiler = StabilizerCompiler()
    metric = Infidelity(target=target_state)
    solver_setting = AlternateTargetSolverSetting(n_iso_graphs=5, n_lc_graphs=5)
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

    solver_setting.monte_carlo_params = {
        "n_sample": 20,
        "map": mcn.McNoiseMap(),
        "compiler": StabilizerCompiler(),
        "seed": 99,
        "n_parallel": 2,
        "n_single": 10,
    }
    solver_setting.monte_carlo = True
    solver = AlternateTargetSolver(
        target=target_state,
        metric=metric,
        compiler=compiler,
        solver_setting=solver_setting,
    )

    example_run = {
        "solver": solver,
        "target": target_state,
        "metric": metric,
        "compiler": compiler,
    }
    io = IO.new_directory(
        folder="benchmarks", include_date=True, include_time=True, include_id=False
    )

    data = benchmark_run_graph_search_solver(example_run, "test_pipeline", io)
    assert type(data) == dict
