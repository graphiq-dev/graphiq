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
"""
Tests that RandomSearchSolver, EvolutionarySolver and HybridEvolutionarySolver each
build their own default settings object instead of sharing one module-level instance.
"""

import networkx as nx
import pytest

from graphiq.backends.density_matrix.compiler import DensityMatrixCompiler
from graphiq.backends.stabilizer.compiler import StabilizerCompiler
from graphiq.backends.stabilizer.functions.rep_conversion import (
    get_clifford_tableau_from_graph,
)
from graphiq.benchmarks.circuits import ghz3_state_circuit
from graphiq.metrics import Infidelity
from graphiq.solvers.evolutionary_solver import (
    EvolutionarySolver,
    EvolutionarySolverSetting,
)
from graphiq.solvers.hybrid_solvers import HybridEvolutionarySolver
from graphiq.solvers.solver_base import RandomSearchSolver, RandomSearchSolverSetting
from graphiq.state import QuantumState

_NO_SETTING = object()


def _build_random_search_solver(solver_setting=_NO_SETTING):
    _, target = ghz3_state_circuit()
    metric = Infidelity(target)
    compiler = DensityMatrixCompiler()
    if solver_setting is _NO_SETTING:
        return RandomSearchSolver(target, metric, compiler)
    return RandomSearchSolver(target, metric, compiler, solver_setting=solver_setting)


def _build_evolutionary_solver(solver_setting=_NO_SETTING):
    _, target = ghz3_state_circuit()
    metric = Infidelity(target)
    compiler = DensityMatrixCompiler()
    if solver_setting is _NO_SETTING:
        return EvolutionarySolver(
            target=target,
            metric=metric,
            compiler=compiler,
            n_emitter=1,
            n_photon=3,
        )
    return EvolutionarySolver(
        target=target,
        metric=metric,
        compiler=compiler,
        n_emitter=1,
        n_photon=3,
        solver_setting=solver_setting,
    )


def _build_hybrid_evolutionary_solver(solver_setting=_NO_SETTING):
    tableau = get_clifford_tableau_from_graph(nx.path_graph(3))
    target = QuantumState(tableau, rep_type="s")
    metric = Infidelity(target)
    compiler = StabilizerCompiler()
    if solver_setting is _NO_SETTING:
        return HybridEvolutionarySolver(target=target, metric=metric, compiler=compiler)
    return HybridEvolutionarySolver(
        target=target,
        metric=metric,
        compiler=compiler,
        solver_setting=solver_setting,
    )


_CASES = [
    pytest.param(
        _build_random_search_solver, RandomSearchSolverSetting, id="RandomSearchSolver"
    ),
    pytest.param(
        _build_evolutionary_solver, EvolutionarySolverSetting, id="EvolutionarySolver"
    ),
    pytest.param(
        _build_hybrid_evolutionary_solver,
        EvolutionarySolverSetting,
        id="HybridEvolutionarySolver",
    ),
]


@pytest.mark.parametrize("build_solver, setting_cls", _CASES)
def test_default_setting_is_per_instance(build_solver, setting_cls):
    a = build_solver()
    b = build_solver()

    assert a.setting is not b.setting

    a.setting.n_stop = 7
    assert b.setting.n_stop == 50

    c = build_solver()
    assert c.setting.n_stop == 50

    s = setting_cls()
    solver = build_solver(solver_setting=s)
    assert solver.setting is s
