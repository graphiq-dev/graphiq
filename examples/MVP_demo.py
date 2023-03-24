"""
Demonstration of MVP capabilities
"""
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

from src.backends.density_matrix.compiler import DensityMatrixCompiler
from src.backends.graph.state import Graph
from src.solvers.evolutionary_solver import EvolutionarySolver
import benchmarks.circuits as circ
from src.metrics import Infidelity
from src.visualizers.density_matrix import density_matrix_heatmap
import src.backends.density_matrix.functions as dmf
import src.backends.state_representation_conversion as sc


def get_graph_equivalent(target_state):
    graph_state = sc.density_to_graph(target_state)
    nx_graph = nx.from_numpy_matrix(graph_state)
    return Graph(nx_graph)


def run_solve(target_function, seed, graph=False):
    # Generate target state
    ideal_circuit, target = target_function()
    target_state = target["dm"]

    # Generate closest graph state
    if graph:
        target_graph_state = get_graph_equivalent(target_state)

    # Create a compiler--this will be used to simulate the circuit and get its output state
    compiler = DensityMatrixCompiler()
    compiler.measurement_determinism = 1

    # Create a Metric function--this is the cost function according to which we judge whether or not a circuit is good
    # Our solver will seek to MINIMIZE this metric
    metric = Infidelity(target_state)

    # Create a setup the solver
    solver = EvolutionarySolver(
        target=target_state,
        metric=metric,
        circuit=None,
        compiler=compiler,
        n_emitter=target["n_emitters"],
        n_photon=target["n_photons"],
    )

    solver.seed(
        seed
    )  # this makes the result replicable (since there is some randomness inherent to the solver
    solver.n_stop = 115
    solver.solve()

    # Grab the best result found by the solver
    assert np.isclose(solver.hof[0][0], 0)
    circuit = solver.hof[0][1]

    # Generate final state
    state = compiler.compile(circuit)
    state_data = dmf.partial_trace(
        state.data,
        keep=list(range(target["n_photons"])),
        dims=(target["n_photons"] + target["n_emitters"]) * [2],
    )

    if graph:
        output_graph = get_graph_equivalent(state_data)

    # Compare the circuit we found to the Li et al. circuit
    if not graph:
        fig, ax = plt.subplots(3, 2, figsize=(10, 10))
    else:
        fig, ax = plt.subplots(4, 2, figsize=(12, 12))
    ideal_circuit.draw_circuit(show=False, ax=ax[0][0])
    circuit.draw_circuit(show=False, ax=ax[0][1])
    ax[0][0].set_title("Li et al. circuit")
    ax[0][1].set_title("Solver circuit")

    # Compare the state we found with the expected state
    density_matrix_heatmap(target_state, axs=[ax[1][0], ax[2][0]])
    density_matrix_heatmap(state_data, axs=[ax[1][1], ax[2][1]])
    ax[1][0].set_title("Target state (real)")
    ax[1][1].set_title("Output state (real)")
    ax[2][0].set_title("Target state (imaginary)")
    ax[2][1].set_title("Output state (imaginary)")

    if graph:
        target_graph_state.draw(show=False, ax=ax[3][0], with_labels=False)
        ax[3][0].set_title("Target state (graph representation)")
        output_graph.draw(show=False, ax=ax[3][1], with_labels=False)
        ax[3][1].set_title("Output state (graph representation)")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    EvolutionarySolver.tournament_k = 0  # indicates no selection

    run_solve(circ.ghz3_state_circuit, 1, graph=False)
    run_solve(circ.ghz4_state_circuit, 1, graph=False)
    run_solve(circ.linear_cluster_3qubit_circuit, 1, graph=True)
    run_solve(circ.linear_cluster_4qubit_circuit, 2, graph=True)
