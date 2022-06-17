"""
Utilities for benchmarking solver performance
"""
import itertools
import pandas as pd
import numpy as np
import time
import os
import datetime
import copy
import json
import matplotlib.pyplot as plt

from src.backends.density_matrix.functions import partial_trace
from src.backends.density_matrix.compiler import DensityMatrixCompiler
from src.solvers.rule_based_random_solver import RuleBasedRandomSearchSolver
import benchmarks.circuits_original as circ
from src.metrics import Infidelity


def circuit_measurement_independent(circuit, compiler):
    # Note: this doesn't check every option, only "all 0s, all 1s"
    original_measurement_determinism = compiler.measurement_determinism

    compiler.measurement_determinism = 0
    state1 = compiler.compile(circuit)
    compiler.measurement_determinism = 1
    state2 = compiler.compile(circuit)
    compiler.measurement_determinism = original_measurement_determinism

    state_data1 = partial_trace(state1.data, keep=list(range(0, circuit.n_quantum - 1)), dims=circuit.n_quantum * [2])
    state_data2 = partial_trace(state2.data, keep=list(range(0, circuit.n_quantum - 1)), dims=circuit.n_quantum * [2])

    return np.allclose(state_data1, state_data2), state_data1, state_data2


def benchmark_data(solver_class, targets, metric_class, compiler, target_type='dm', per_target_retries=1, seed_offset=0,
                   save_directory=None):
    """
    Benchmarks a solver performance on a series of target circuits/states

    Saves and returns solver information (e.g. n_pop, etc.), compute information, a benchmark results dataframe, a list
    containing the best circuit of each target)

    Benchmark results should have:
        1. Each column is a solver run (which columns being grouped by their target circuit as well)
        2. Each row holds information. Current targets for implementation include:
            a) Runtime of solver iteration [not done]
            b) loss function performance reported (float) [not done]
            b.1) Circuit (only useful if we're not saving the dataframe)
            c) Whether or not the measurement results impact the final state (bool) [not done]
            d) Final circuit depth [not done]
            e) Runtime of solver iteration [not done]

    :param solver_class: the solver being benchmarked
    :type solver_class: SolverBase
    :param targets: a list of (target_circuit, target_state) on which to evaluate
    :type targets: list
    :param metric_class: the class of the cost function metric to use in benchmarking
    :type metric_class: MetricBase
    :param compiler: the compiler which the solver should use
    :param target_type: a string describing the target state representation
    :type target_type: str
    :param per_target_retries: the number of times we should reseed and run the same solver problem
    :type per_target_retries: int
    :param save_directory: if None, return data without saving. Else, save data to specified directory
    :type save_directory: str
    :return: dataframe summarizing benchmark information + other info
    :rtype: tuple
    """

    # TODO: save solver data
    # TODO: save compute data
    # TODO: log times for each solver run
    # TODO: save metric choice

    # Create pandas dataframe
    target_state_names = [target['name'] for (circuit, target) in targets]
    index_tuples = itertools.product(target_state_names, [i for i in range(per_target_retries)])
    column_index = pd.MultiIndex.from_tuples(index_tuples, names=['Target state name', 'Retry #'])
    data_fields = ['runtime (s)', f'{metric_class.__name__} score', 'Circuit', 'Measurement independent (T/F)',
                   'Circuit depth']

    df = pd.DataFrame(index=data_fields, columns=column_index)
    hof_circuits = []

    for ideal_circuit, target in targets:
        # TODO: this is not flexible to the type of metric being used, should be adapted later
        target_state = target[target_type]

        for i in range(per_target_retries):
            # We remake the solver to avoid awkward issues with improper updates
            solver = solver_class(target=target_state, metric=metric_class(target_state), compiler=compiler,
                                  n_emitter=target['n_emitters'],
                                  n_photon=target['n_photons'])
            start_time = time.time()
            solver.solve(i + seed_offset)
            total_time = time.time() - start_time
            df.loc['runtime (s)', (target['name'], i)] = total_time
            df.loc[f'{metric_class.__name__} score', (target['name'], i)] = copy.deepcopy(solver.hof[0][0])
            df.loc[f'Circuit', (target['name'], i)] = copy.deepcopy(solver.hof[1][0])
            df.loc[f'Measurement independent (T/F)', (target['name'], i)] = \
                copy.deepcopy(circuit_measurement_independent(solver.hof[1][0], compiler)[0])
            df.loc[f'Circuit depth', (target['name'], i)] = 'Coming soon...'  # TODO: implement a circuit class func

            circuit_data = {
                'name': target['name'],
                'seed': i,
                'runtime (s)': total_time,
                'Measurement independent (T/F)': df.loc[f'Measurement independent (T/F)', (target['name'], i)],
                'Circuit description': solver.hof[1][0].to_openqasm()
            }
            hof_circuits.append(circuit_data)

    if save_directory is not None:
        dir_name = os.path.join(save_directory, f'solver_benchmark_{datetime.datetime.now().strftime("%Y_%m_%d-%H-%M-%S")}')
        os.mkdir(dir_name)
        df.to_csv(f'{dir_name}/benchmark_results.csv')
        for i in range(len(hof_circuits)):
            with open(f"{dir_name}/circuit_{i}.json", "w") as fp:
                json.dump(hof_circuits[i], fp)

            with open(f"{dir_name}/circuit_{i}_description.txt", 'w') as f:
                f.write(hof_circuits[i]['Circuit description'])

        for _, target in targets:
            for i in range(per_target_retries):
                print()
                df.loc[f'Circuit', (target['name'], i)].draw_circuit(show=False)
                plt.savefig(f'{dir_name}/circuit_{target["name"]}_{i}.png')

    return df


if __name__ == "__main__":
    target_list = [circ.ghz3_state_circuit(), circ.ghz4_state_circuit(), circ.linear_cluster_3qubit_circuit(),
                   circ.linear_cluster_4qubit_circuit()]
    df = benchmark_data(RuleBasedRandomSearchSolver, target_list, Infidelity, DensityMatrixCompiler(),
                        per_target_retries=10, seed_offset=0, save_directory='./save_benchmarks/')
