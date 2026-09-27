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
import functools
import json
import runpy

from graphiq.backends.density_matrix.compiler import DensityMatrixCompiler
from graphiq.benchmarks.benchmark_utils import benchmark_data
from graphiq.io import IO
from graphiq.metrics import Infidelity
from graphiq.solvers.evolutionary_solver import (
    EvolutionarySolver,
    EvolutionarySolverSetting,
)
from graphiq.benchmarks.circuits import ghz3_state_circuit


def test_one_benchmarking_run():
    # define a single run by creating an instance of each object
    _, target = ghz3_state_circuit()
    compiler = DensityMatrixCompiler()
    metric = Infidelity(target=target)

    solver1 = EvolutionarySolver(
        target=target,
        metric=metric,
        compiler=compiler,
        n_photon=3,
        n_emitter=1,
        solver_setting=EvolutionarySolverSetting(n_stop=10, n_pop=5),
    )

    example_run1 = {
        "solver": solver1,
        "target": target,
        "metric": metric,
        "compiler": compiler,
    }

    # put all runs into a dict, with a descriptive name
    runs = {
        "example_run1": example_run1,
    }

    io = IO.new_directory(
        folder="test_benchmarks",
        include_date=False,
        include_time=False,
        include_id=False,
    )
    # df = benchmark(runs=runs, io=io, remote=False)
    # print(df)


def _only_benchmark_folder(path):
    (folder,) = path.joinpath("benchmarks").glob("*_solver-benchmarks_*")
    return folder


def test_benchmark_data_saves_its_results(tmp_path, monkeypatch):
    monkeypatch.setattr(IO, "default_path", tmp_path)
    circuit, state = ghz3_state_circuit()
    target = {
        "name": "ghz3",
        "dm": state,
        "n_emitters": circuit.n_emitters,
        "n_photons": circuit.n_photons,
    }
    compiler = DensityMatrixCompiler()
    compiler.measurement_determinism = 1
    setting = EvolutionarySolverSetting(n_stop=2, n_pop=4)

    solver_data, _, hof_circuits = benchmark_data(
        functools.partial(EvolutionarySolver, solver_setting=setting),
        [(circuit, target)],
        Infidelity,
        compiler,
        seed_offset=3,
        save_directory="benchmarks",
    )

    folder = _only_benchmark_folder(tmp_path)
    with open(folder.joinpath("solver_info.json")) as file:
        solver_info = json.load(file)
    assert solver_info["solver setting"] == str(setting)
    others = {
        key: value for key, value in solver_data.items() if key != "solver setting"
    }
    assert {
        key: value for key, value in solver_info.items() if key != "solver setting"
    } == json.loads(json.dumps(others))

    with open(folder.joinpath("circuit_target0_retry0.json")) as file:
        circuit_info = json.load(file)
    assert set(circuit_info) == set(hof_circuits[0])
    assert circuit_info["name"] == "ghz3"
    assert circuit_info["seed"] == 3
    assert circuit_info["Circuit description"] == hof_circuits[0]["Circuit description"]

    # the returned solver info still holds the setting object
    assert solver_data["solver setting"] is setting


def test_the_benchmark_utils_script_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(IO, "default_path", tmp_path)

    runpy.run_module("graphiq.benchmarks.benchmark_utils", run_name="__main__")

    folder = _only_benchmark_folder(tmp_path)
    assert folder.joinpath("solver_info.json").is_file()
    assert folder.joinpath("circuit_target0_retry0.json").is_file()


if __name__ == "__main__":
    test_one_benchmarking_run()
