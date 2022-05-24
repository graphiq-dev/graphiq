"""
Examples of defining and simulating quantum circuits for a variety of small quantum states
"""
import numpy as np
import matplotlib.pyplot as plt

from src.circuit import CircuitDAG
from src.ops import *
from src.backends.density_matrix.compiler import DensityMatrixCompiler

from src.backends.density_matrix.functions import ketz0_state, ketz1_state, tensor, ket2dm, partial_trace, fidelity
from src.visualizers.density_matrix import density_matrix_bars

from src.libraries.circuits import bell_state_circuit, ghz3_state_circuit, ghz4_state_circuit, \
    linear_cluster_3qubit_circuit, linear_cluster_4qubit_circuit


if __name__ == "__main__":
    # example_circuit = bell_state_circuit
    # example_circuit = ghz3_state_circuit
    # example_circuit = ghz4_state_circuit
    # example_circuit = linear_cluster_3qubit_circuit
    example_circuit = linear_cluster_4qubit_circuit

    circuit, ideal_state = example_circuit()

    compiler = DensityMatrixCompiler()
    state = compiler.compile(circuit)

    # trace out the ancilla qubit (note, comment this out if running the Bell-state circuit
    state = partial_trace(state.data, keep=list(range(0, circuit.n_quantum-1)), dims=circuit.n_quantum * [2])

    f = fidelity(state, ideal_state['dm'])
    print(f"Fidelity with the ideal state is {f}")

    fig, axs = density_matrix_bars(ideal_state['dm'])
    fig.suptitle("Ideal density matrix")

    fig, axs = density_matrix_bars(state)
    fig.suptitle("Simulated circuit density matrix")

    plt.show()
