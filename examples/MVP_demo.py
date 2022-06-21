"""
Demonstration of MVP capabilities
"""
import numpy as np
import matplotlib.pyplot as plt

from src.backends.density_matrix.compiler import DensityMatrixCompiler
from src.solvers.rule_based_random_solver import RuleBasedRandomSearchSolver
import benchmarks.circuits_original as circ
from src.metrics import Infidelity
from src.visualizers.density_matrix import density_matrix_heatmap
import src.backends.density_matrix.functions as dmf


def run_solve(target_function, seed):
    # Create a compiler--this will be used to simulate the circuit and get its output state
    compiler = DensityMatrixCompiler()
    compiler.measurement_determinism = 1

    # Generate target state
    ideal_circuit, target = target_function()
    target_state = target['dm']

    # Create a Metric function--this is the cost function according to which we judge whether or not a circuit is good
    # Our solver will seek to MINIMIZE this metric
    metric = Infidelity(target_state)

    # Create a setup the solver
    solver = RuleBasedRandomSearchSolver(target=target_state, metric=metric, compiler=compiler,
                                         n_emitter=target['n_emitters'], n_photon=target['n_photons'],
                                         selection_active=False)

    solver.seed(seed)  # this makes the result replicable (since there is some randomness inherent to the solver
    solver.n_stop = 115
    solver.solve()

    # Grab the best result found by the solver
    assert np.isclose(solver.hof[0][0], 0)
    circuit = solver.hof[0][1]

    # Generate final state
    state = compiler.compile(circuit)
    state_data = dmf.partial_trace(state.data,
                                   keep=list(range(target['n_photons'])),
                                   dims=(target['n_photons'] + target['n_emitters']) * [2])

    # Compare the circuit we found to the Li et al. circuit
    fig, ax = plt.subplots(3, 2, figsize=(10, 10))
    ideal_circuit.draw_circuit(show=False, ax=ax[0][0])
    circuit.draw_circuit(show=False, ax=ax[0][1])
    ax[0][0].set_title('Li et al. circuit')
    ax[0][1].set_title('Solver circuit')

    # Compare the state we found with the expected state
    density_matrix_heatmap(target_state, axs=ax[1])
    density_matrix_heatmap(state_data, axs=ax[2])
    ax[1][0].set_title('Target state (real)')
    ax[1][1].set_title('Target state (imaginary)')
    ax[2][0].set_title('Simulated state (real)')
    ax[2][1].set_title('Simulated state (imaginary)')

    plt.tight_layout()
    plt.show()


if __name__=='__main__':
    run_solve(circ.ghz3_state_circuit, 0)
    run_solve(circ.ghz4_state_circuit, 0)
    run_solve(circ.linear_cluster_3qubit_circuit, 0)
    run_solve(circ.linear_cluster_4qubit_circuit, 0)


