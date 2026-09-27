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
import json

import networkx as nx

import graphiq.circuit.ops as ops
from graphiq.circuit.circuit_dag import CircuitDAG
from graphiq.data_collection.correlation_module import graph_to_circ, graph_to_cnot
from graphiq.data_collection.user_interface import find_best, result_maker, rgs_analysis
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


def test_rgs_analysis_saves_n_cnots_without_an_empty_n_cnot_column(
    tmp_path, monkeypatch
):
    ui = "graphiq.data_collection.user_interface"
    monkeypatch.setattr(f"{ui}.correlation_checker", lambda *a, **k: None)
    monkeypatch.setattr(f"{ui}.plot_figs", lambda *a, **k: None)
    graph = nx.cycle_graph(5)
    result = SolverResult([graph_to_circ(graph)], ["g"])
    result["g"] = [graph]
    (tmp_path / "RGS" / "run").mkdir(parents=True)

    rgs_analysis(result, str(tmp_path), "run")

    saved = json.loads((tmp_path / "RGS" / "run" / "run.json").read_text())
    assert "n_cnot" not in saved
    assert saved["n_cnots"] == [graph_to_cnot(graph)]
    assert "n_cnots" in result._properties
    assert "n_cnot" not in result._properties


def test_rgs_analysis_creates_its_save_directory(tmp_path, monkeypatch):
    ui = "graphiq.data_collection.user_interface"
    monkeypatch.setattr(f"{ui}.correlation_checker", lambda *a, **k: None)
    monkeypatch.setattr(f"{ui}.plot_figs", lambda *a, **k: None)
    graph = nx.cycle_graph(5)
    result = SolverResult([graph_to_circ(graph)], ["g"])
    result["g"] = [graph]

    rgs_analysis(result, str(tmp_path), "run")

    saved = json.loads((tmp_path / "RGS" / "run" / "run.json").read_text())
    assert saved["n_cnots"] == [graph_to_cnot(graph)]
