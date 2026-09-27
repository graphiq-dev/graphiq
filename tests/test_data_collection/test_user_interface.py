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

import graphiq.circuit.ops as ops
from graphiq.circuit.circuit_dag import CircuitDAG
from graphiq.data_collection.user_interface import find_best, result_maker
from graphiq.metrics import CircuitUnitaryCount
from graphiq.solvers.solver_result import SolverResult


def test_result_maker_counts_each_one_qubit_gate_once():
    circuit = CircuitDAG(n_emitter=1, n_photon=0, n_classical=0)
    for gate in [ops.SigmaX, ops.SigmaY, ops.SigmaZ, ops.PhaseDagger, ops.Hadamard]:
        circuit.add(gate(register=0, reg_type="e"))

    result = result_maker(
        SolverResult([circuit]), graph_met_list=[], circ_met_list=["n_unitary"]
    )

    assert result["n_unitary"] == [5]
    assert result["n_unitary"][0] == CircuitUnitaryCount().evaluate(None, circuit)


def test_find_best_without_dir_path_returns_results_and_writes_nothing(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    out = find_best(nx.to_numpy_array(nx.path_graph(3)), n_reordering=1, conf=0)

    assert len(out["cost"]) >= 1
    assert list(tmp_path.iterdir()) == []
