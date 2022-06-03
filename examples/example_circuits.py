"""
Examples of defining and simulating quantum circuits for a variety of small quantum states
"""
import matplotlib.pyplot as plt

from src.backends.density_matrix.compiler import DensityMatrixCompiler

from src.backends.density_matrix.functions import partial_trace, fidelity
from src.visualizers.density_matrix import density_matrix_bars

from benchmarks.circuits import *


if __name__ == "__main__":
    # example_circuit = bell_state_circuit
    # example_circuit = ghz3_state_circuit
    example_circuit = ghz4_state_circuit
    # example_circuit = linear_cluster_3qubit_circuit
    # example_circuit = linear_cluster_4qubit_circuit

    circuit, ideal_state = example_circuit()

    # Visualize
    circuit.draw_dag()  # DAG visualization
    circuit.draw_circuit()  # circuit visualization (qiskit visualizer)
    circuit.validate()
    # Compile
    compiler = DensityMatrixCompiler()
    state = compiler.compile(circuit)

    if example_circuit is bell_state_circuit:
        state_data = state.data
    else:
        # trace out the ancilla qubit
        state_data = partial_trace(state.data, keep=list(range(0, circuit.n_quantum-1)), dims=circuit.n_quantum * [2])

    f = fidelity(state_data, ideal_state['dm'])
    print(f"Fidelity with the ideal state is {f}")

    fig, axs = density_matrix_bars(ideal_state['dm'])
    fig.suptitle("Ideal density matrix")

    fig, axs = density_matrix_bars(state_data)
    fig.suptitle("Simulated circuit density matrix")

    plt.show()
