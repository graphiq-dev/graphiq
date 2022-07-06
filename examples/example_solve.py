"""
Example of using a solver to discover circuits to generate a target state
"""
import matplotlib.pyplot as plt
import time

from src.backends.density_matrix.compiler import DensityMatrixCompiler
from src.solvers.evolutionary_solver import EvolutionarySolver
from src.metrics import Infidelity
from src.circuit import CircuitDAG
import src.backends.density_matrix.functions as dmf

from src.visualizers.density_matrix import density_matrix_bars
from benchmarks.circuits import bell_state_circuit


if __name__ == "__main__":
    # %% here we have access
    EvolutionarySolver.n_stop = 40
    EvolutionarySolver.n_pop = 150
    EvolutionarySolver.n_hof = 10
    EvolutionarySolver.tournament_k = 10

    # %% comment/uncomment for reproducibility
    # RuleBasedRandomSearchSolver.seed(1)

    # %% select which state we want to target
    from benchmarks.circuits import *

    circuit_ideal, state_ideal = linear_cluster_3qubit_circuit()

    # %% construct all of our important objects
    target = state_ideal["dm"]
    compiler = DensityMatrixCompiler()
    metric = Infidelity(target=target)

    n_photon = 3
    n_emitter = 1
    solver = EvolutionarySolver(
        target=target,
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

    state_data = dmf.partial_trace(
        state.data, keep=list(range(n_photon)), dims=(n_photon + n_emitter) * [2]
    )

    # extract the best performing circuit
    fig, axs = density_matrix_bars(target)
    fig.suptitle("Target density matrix")
    plt.show()

    fig, axs = density_matrix_bars(state_data)
    fig.suptitle("Simulated density matrix")
    plt.show()

    circuit.draw_circuit()
