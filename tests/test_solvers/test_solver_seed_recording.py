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
import pytest

from graphiq.backends.density_matrix.compiler import DensityMatrixCompiler
from graphiq.backends.stabilizer.compiler import StabilizerCompiler
from graphiq.backends.stabilizer.functions.rep_conversion import (
    get_clifford_tableau_from_graph,
)
from graphiq.benchmarks.circuits import ghz3_state_circuit
from graphiq.benchmarks.graph_states import linear_cluster_state
from graphiq.metrics import Infidelity
from graphiq.solvers.evolutionary_solver import EvolutionarySolver
from graphiq.solvers.solver_base import SolverBase
from graphiq.solvers.time_reversed_solver import TimeReversedSolver
from graphiq.state import QuantumState


@pytest.fixture(autouse=True)
def _restore_last_seed():
    saved = getattr(SolverBase, "last_seed", None)
    yield
    SolverBase.last_seed = saved


def _evolutionary_solver():
    _, ideal_state = ghz3_state_circuit()
    return EvolutionarySolver(
        target=ideal_state,
        metric=Infidelity(target=ideal_state),
        compiler=DensityMatrixCompiler(),
        n_emitter=1,
        n_photon=3,
    )


def _time_reversed_solver():
    tableau = get_clifford_tableau_from_graph(linear_cluster_state(2).data)
    target = QuantumState(tableau, rep_type="stab")
    return TimeReversedSolver(
        target=target, metric=Infidelity(target=target), compiler=StabilizerCompiler()
    )


@pytest.mark.parametrize("build", [_evolutionary_solver, _time_reversed_solver])
def test_solver_reports_its_seed(build):
    solver = build()
    solver.seed(4)
    assert solver.solver_info["seed"] == 4


def test_class_level_seed_is_visible_to_an_instance():
    EvolutionarySolver.seed(7)
    assert _evolutionary_solver().solver_info["seed"] == 7
