"""
Examples of defining and simulating quantum circuits for a variety of small quantum states
"""
import matplotlib.pyplot as plt

from graphiq.backends.density_matrix.compiler import DensityMatrixCompiler

from graphiq.metrics import Infidelity
from graphiq.visualizers.density_matrix import density_matrix_bars

if __name__ == "__main__":
    # example_circuit = bell_state_circuit
    # example_circuit = ghz3_state_circuit
    # example_circuit = ghz4_state_circuit
    # example_circuit = linear_cluster_3qubit_circuit
    example_circuit = linear_cluster_4qubit_circuit

    circuit, ideal_state = example_circuit()

    # Visualize
    circuit.draw_dag()  # DAG visualization
    circuit.draw_circuit()  # circuit visualization (qiskit visualizer)
    circuit.validate()
    # Compile
    compiler = DensityMatrixCompiler()
    state = compiler.compile(circuit)
    metric = Infidelity(ideal_state)

    if example_circuit is not bell_state_circuit:
        # trace out the ancilla qubit
        state.partial_trace(
            keep=[*range(0, circuit.n_quantum - 1)], dims=circuit.n_quantum * [2]
        )

    infid = metric.evaluate(state, circuit)
    print(f"Infidelity with the ideal state is {infid}")

    fig, _ = density_matrix_bars(ideal_state.rep_data.data)
    fig.suptitle("Ideal density matrix")

    fig, _ = density_matrix_bars(state.rep_data.data)
    fig.suptitle("Simulated circuit density matrix")

    plt.show()
