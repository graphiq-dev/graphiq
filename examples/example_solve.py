"""
Example of using a solver to discover circuits to generate a target state
"""
import matplotlib.pyplot as plt
import time

from graphiq.backends.density_matrix.compiler import DensityMatrixCompiler
from graphiq.solvers.evolutionary_solver import EvolutionarySolver
from graphiq.metrics import Infidelity
import graphiq.backends.density_matrix.functions as dmf

from benchmarks.circuits import *
from graphiq.visualizers.density_matrix import density_matrix_bars
from graphiq.visualizers.solver_logs import plot_solver_logs

if __name__ == "__main__":
    # %% here we have access
    EvolutionarySolver.n_stop = 40
    EvolutionarySolver.n_pop = 150
    EvolutionarySolver.n_hof = 10
    EvolutionarySolver.tournament_k = 10

    # %% comment/uncomment for reproducibility
    EvolutionarySolver.seed(3)

    # %% select which state we want to target

    circuit_ideal, target_state = linear_cluster_3qubit_circuit()

    # %% construct all of our important objects
    compiler = DensityMatrixCompiler()
    metric = Infidelity(target=target_state)

    n_photon = 3
    n_emitter = 1
    solver = EvolutionarySolver(
        target=target_state,
        metric=metric,
        compiler=compiler,
        n_photon=n_photon,
        n_emitter=n_emitter,
    )

    # %% call the solver.solve() function to implement the random search algorithm
    t0 = time.time()
    solver.solve()
    t1 = time.time()

    # %% print/plot the results
    print(solver.hof)
    print(f"Total time {t1 - t0}")

    circuit = solver.hof[0][1]
    state = compiler.compile(circuit)  # this will pass out a density matrix object

    state.partial_trace(keep=[*range(n_photon)], dims=(n_photon + n_emitter) * [2])

    # extract the best performing circuit
    fig, axs = density_matrix_bars(target_state.dm.data)
    fig.suptitle("Target density matrix")
    plt.show()

    fig, axs = density_matrix_bars(state.dm.data)
    fig.suptitle("Simulated density matrix")
    plt.show()

    circuit.draw_circuit()

    fig, axs = plot_solver_logs(solver.logs)
    plt.show()
