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
Example script for benchmarking multiple solvers
"""

from graphiq.backends.density_matrix.compiler import DensityMatrixCompiler
from graphiq.solvers.evolutionary_solver import EvolutionarySolver
from graphiq.metrics import Infidelity
from graphiq.io import IO
from graphiq.visualizers.solver_logs import plot_solver_logs
from graphiq.benchmarks.circuits import ghz3_state_circuit, ghz4_state_circuit
from graphiq.benchmarks.pipeline import benchmark, run_combinations


if __name__ == "__main__":
    # %% provide all combinations of solvers, targets, compilers, and metrics to run as a list
    solvers = [
        (EvolutionarySolver, dict(n_stop=30, n_pop=80, n_hof=10, tournament_k=2)),
        (EvolutionarySolver, dict(n_stop=30, n_pop=100, n_hof=10, tournament_k=2)),
    ]

    # provide a list of targets
    targets = [
        (ghz3_state_circuit()[1]["dm"], dict(name="ghz3", n_photon=3, n_emitter=1)),
        (ghz4_state_circuit()[1]["dm"], dict(name="ghz4", n_photon=4, n_emitter=1)),
    ]

    compilers = [DensityMatrixCompiler]

    metrics = [Infidelity]

    # take all combinations of the lists provided above
    runs = run_combinations(solvers, targets, compilers, metrics)

    io = IO.new_directory(
        folder="benchmarks", include_date=True, include_time=True, include_id=False
    )
    df = benchmark(runs=runs, io=io, remote=True)
    print(df)

    plot_solver_logs()
