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
Utilities for benchmarking solver performance
"""

import itertools
import pandas as pd
import numpy as np
import time
import copy

from graphiq.backends.density_matrix.compiler import DensityMatrixCompiler
from graphiq.solvers.evolutionary_solver import EvolutionarySolver
from graphiq.benchmarks import circuits as circ
from graphiq.metrics import Infidelity

from graphiq.io import IO


def circuit_measurement_independent(circuit, compiler):
    """
    Check whether the compiled state, after tracing out the last qubit, is the same regardless
    of which classical measurement outcome is forced during compilation.

    Compiles the circuit twice, once with the compiler's measurement determinism forced to 0
    and once forced to 1, then restores the compiler's original measurement_determinism
    setting. Only compares the two deterministic ("all 0s", "all 1s") resolutions, not every
    possible measurement outcome, and only works with a density-matrix compiler/state at the
    moment.

    :param circuit: the circuit to compile
    :type circuit: CircuitDAG
    :param compiler: the compiler to compile it with; its measurement_determinism attribute is
        overwritten twice and then restored
    :type compiler: DensityMatrixCompiler
    :return: whether the two traced-out states are numerically close, and the two traced-out
        states themselves
    :rtype: tuple[bool, QuantumState, QuantumState]
    """
    # Note: this doesn't check every option, only "all 0s, all 1s"
    # TODO: make applicable to other representations. This only works on density matrices at the moment
    original_measurement_determinism = compiler.measurement_determinism

    compiler.measurement_determinism = 0
    state1 = compiler.compile(circuit)
    compiler.measurement_determinism = 1
    state2 = compiler.compile(circuit)
    compiler.measurement_determinism = original_measurement_determinism

    state1.partial_trace(
        keep=list(range(0, circuit.n_quantum - 1)),
        dims=circuit.n_quantum * [2],
    )
    state2.partial_trace(
        keep=list(range(0, circuit.n_quantum - 1)),
        dims=circuit.n_quantum * [2],
    )

    return np.allclose(state1.rep_data.data, state2.rep_data.data), state1, state2


def benchmark_data(
    solver_class,
    targets,
    metric_class,
    compiler,
    target_type="dm",
    per_target_retries=1,
    seed_offset=0,
    save_directory=None,
):
    """
    Benchmarks a solver performance on a series of target circuits/states

    Saves and returns solver information (e.g. n_pop, etc.), compute information, a benchmark results dataframe, a list
    containing the best circuit of each target)

    Benchmark results should have:
        1. Each column is a solver run (which columns being grouped by their target circuit as well)
        2. Each row holds information. Current targets for implementation include:
            a) Runtime of solver iteration [not done]
            b) loss function performance reported (float) [not done]
                1) Circuit (only useful if we're not saving the dataframe)
            c) Whether or not the measurement results impact the final state (bool) [not done]
            d) Final circuit depth [not done]
            e) Runtime of solver iteration [not done]

    :param solver_class: the solver being benchmarked: a class, or any callable returning a solver,
        that is called with the keyword arguments target, metric, compiler, n_emitter and n_photon
        (the last two from the target's "n_emitters" and "n_photons"). The solver must have a hall
        of fame, hof, after solve(), whose first entry is the best (score, circuit) pair.
        EvolutionarySolver qualifies (use functools.partial to give it a solver_setting). A class
        that works out n_emitter and n_photon itself, such as HybridEvolutionarySolver, can be
        passed through a callable that drops them, e.g.
        lambda n_emitter, n_photon, **kwargs: HybridEvolutionarySolver(**kwargs); for
        HybridEvolutionarySolver this works only for targets whose density matrix is itself a
        graph state (not the GHZ targets in graphiq.benchmarks.circuits, for example).
        TimeReversedSolver has no hof and cannot be benchmarked.
    :type solver_class: type[SolverBase] or callable
    :param targets: a list of (target_circuit, target) pairs on which to evaluate; target_circuit
        is not read, and target is a dict holding "name", "n_emitters", "n_photons" and the target
        state under the key target_type
    :type targets: list[tuple]
    :param metric_class: the class of the cost function metric to use in benchmarking
    :type metric_class: MetricBase
    :param compiler: the compiler which the solver should use; it must be a density-matrix
        compiler, since each best circuit is also checked with circuit_measurement_independent,
        which compares density matrices (a StabilizerCompiler raises TypeError there)
    :type compiler: DensityMatrixCompiler
    :param target_type: a string describing the target state representation
    :type target_type: str
    :param per_target_retries: the number of times we should reseed and run the same solver problem
    :type per_target_retries: int
    :param seed_offset: added to each retry's index to form the seed passed to the solver
    :type seed_offset: int
    :param save_directory: if None, return data without saving. Else, save data to specified directory
    :type save_directory: str
    :return: dataframe summarizing benchmark information + other info
    :rtype: tuple
    """
    # TODO: refactor to save as we go if we DO define a save directory,
    #  there's no point on retaking all the data if something fails...

    # TODO: save solver data
    # TODO: save compute data
    if save_directory is not None:
        folder_name = f"solver-benchmarks"
        io = IO.new_directory(
            path=IO.default_path.joinpath(save_directory),
            folder=folder_name,
            include_date=True,
            include_time=True,
            include_id=True,
        )

    # Create pandas dataframe
    target_state_names = [target["name"] for (circuit, target) in targets]
    index_tuples = itertools.product(
        target_state_names, [i for i in range(per_target_retries)]
    )
    column_index = pd.MultiIndex.from_tuples(
        index_tuples, names=["Target state name", "Retry #"]
    )
    data_fields = [
        "runtime (s)",
        f"{metric_class.__name__} score",
        "Circuit",
        "Measurement independent (T/F)",
        "Circuit depth",
    ]

    df = pd.DataFrame(index=data_fields, columns=column_index)
    hof_circuits = []

    for j, (ideal_circuit, target) in enumerate(targets):
        # TODO: this is not flexible to the type of metric being used, should be adapted later
        target_state = target[target_type]

        for i in range(per_target_retries):
            # We remake the solver to avoid awkward issues with improper updates
            solver = solver_class(
                target=target_state,
                metric=metric_class(target_state),
                compiler=compiler,
                n_emitter=target["n_emitters"],
                n_photon=target["n_photons"],
            )
            # seed before reading solver_info, whose "seed" field reports it
            solver.seed(i + seed_offset)
            solver_data = solver.solver_info
            if save_directory and i == 0 and j == 0:
                # a solver setting object is not JSON serializable; save its description
                io.save_json(
                    {
                        key: str(value) if key == "solver setting" else value
                        for key, value in solver_data.items()
                    },
                    "solver_info.json",
                )

            start_time = time.time()
            solver.solve()
            total_time = time.time() - start_time
            df.loc["runtime (s)", (target["name"], i)] = total_time
            df.loc[f"{metric_class.__name__} score", (target["name"], i)] = (
                copy.deepcopy(solver.hof[0][0])
            )
            df.loc[f"Circuit", (target["name"], i)] = copy.deepcopy(solver.hof[0][1])
            df.loc[f"Measurement independent (T/F)", (target["name"], i)] = (
                copy.deepcopy(
                    circuit_measurement_independent(solver.hof[0][1], compiler)[0]
                )
            )
            df.loc[f"Circuit depth", (target["name"], i)] = solver.hof[0][1].depth

            circuit_data = {
                "name": target["name"],
                "seed": i + seed_offset,
                "runtime (s)": total_time,
                "Measurement independent (T/F)": df.loc[
                    f"Measurement independent (T/F)", (target["name"], i)
                ],
                "Circuit description": solver.hof[0][1].to_openqasm(),
            }
            hof_circuits.append(circuit_data)
            if save_directory is not None:
                io.save_json(circuit_data, f"circuit_target{j}_retry{i}.json")
                io.save_txt(
                    circuit_data["Circuit description"],
                    f"circuit_description_target{j}_retry{i}.txt",
                )

    if save_directory is not None:
        io.save_dataframe(df, filename="benchmark_results.csv")

        for _, target in targets:
            for i in range(per_target_retries):
                print()
                fig, axs = df.loc[f"Circuit", (target["name"], i)].draw_circuit(
                    show=False
                )

                io.save_figure(fig, f"circuit_{target['name']}_{i}.png")

    return solver_data, df, hof_circuits


if __name__ == "__main__":
    circuit, state = circ.ghz3_state_circuit()
    target_list = [
        (
            circuit,
            {
                "name": "ghz3",
                "dm": state,
                "n_emitters": circuit.n_emitters,
                "n_photons": circuit.n_photons,
            },
        )
    ]
    compiler = DensityMatrixCompiler()
    compiler.measurement_determinism = 1
    df = benchmark_data(
        EvolutionarySolver,
        target_list,
        Infidelity,
        compiler,
        per_target_retries=1,
        seed_offset=0,
        save_directory="benchmarks",
    )
