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
from graphiq.backends.density_matrix.compiler import DensityMatrixCompiler
from graphiq.io import IO
from graphiq.metrics import Infidelity
from graphiq.solvers.evolutionary_solver import EvolutionarySolver
from graphiq.benchmarks.circuits import ghz3_state_circuit


def test_one_benchmarking_run():
    # define a single run by creating an instance of each object
    _, target = ghz3_state_circuit()
    compiler = DensityMatrixCompiler()
    metric = Infidelity(target=target)

    solver1 = EvolutionarySolver(
        target=target, metric=metric, compiler=compiler, n_photon=3, n_emitter=1
    )
    solver1.n_stop = 10
    solver1.n_pop = 5

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


if __name__ == "__main__":
    test_one_benchmarking_run()
