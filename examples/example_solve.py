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
    EvolutionarySolver.seed(1)

    # %% select which state we want to target
    from benchmarks.circuits import *
    circuit_ideal, state_ideal = linear_cluster_3qubit_circuit()

    # %% construct all of our important objects
    target = state_ideal['dm']
    compiler = DensityMatrixCompiler()
    metric = Infidelity(target=target)

    n_photon = 3
    n_emitter = 1
    solver = EvolutionarySolver(target=target, metric=metric, compiler=compiler,
                                n_photon=n_photon, n_emitter=n_emitter)

    # %% call the solver.solve() function to implement the random search algorithm
    t0 = time.time()
    solver.solve()
    t1 = time.time()

    # %% print/plot the results
    print(solver.hof)
    print(f"Total time {t1 - t0}")

    circuit = solver.hof[0][1]
    state = compiler.compile(circuit)  # this will pass out a density matrix object

    state_data = dmf.partial_trace(state.data,
                                   keep=list(range(n_photon)),
                                   dims=(n_photon + n_emitter) * [2])

    # extract the best performing circuit
    fig, axs = density_matrix_bars(target)
    fig.suptitle("Target density matrix")
    plt.show()

    fig, axs = density_matrix_bars(state_data)
    fig.suptitle("Simulated density matrix")
    plt.show()

    circuit.draw_circuit()

    #%%
    fig, axs = plt.subplots(nrows=2, ncols=2, sharey='row', sharex="col")
    colors = ["teal", "orange"]
    for col, (log_name, log) in enumerate(solver.logs.items()):
        c = colors[col]
        axs[0, col].plot(log['iteration'], log["cost_mean"], color=c, label=f"{log_name}, mean")
        axs[0, col].fill_between(log['iteration'], log["cost_min"], log["cost_max"], color=c, alpha=0.3, label=f"{log_name}, range")

        axs[1, col].plot(log['iteration'], log["depth_mean"], color=c)
        axs[1, col].fill_between(log['iteration'], log["depth_min"], log["depth_max"], color=c, alpha=0.3)

        axs[0, col].set(title=f"{log_name}")

    axs[0, 0].set(ylabel="Cost value")
    axs[1, 0].set(ylabel="Circuit depth")
    axs[1, 0].set(xlabel="Iteration")
    axs[1, 1].set(xlabel="Iteration")

    plt.show()
