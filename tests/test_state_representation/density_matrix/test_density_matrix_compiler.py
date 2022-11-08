from tests.test_flags import visualization

import matplotlib.pyplot as plt

from src.backends.density_matrix import numpy as np
from src.backends.density_matrix.compiler import DensityMatrixCompiler
from src.backends.density_matrix.functions import partial_trace, fidelity
from benchmarks.circuits import (
    ghz3_state_circuit,
    bell_state_circuit,
    ghz4_state_circuit,
)
from src.visualizers.density_matrix import density_matrix_bars
from src.circuit import CircuitDAG
import src.ops as ops


def test_bell_circuit():
    circuit, ideal_state = bell_state_circuit()

    compiler = DensityMatrixCompiler()
    state = compiler.compile(circuit)

    f = fidelity(state.dm.data, ideal_state.dm.data)

    assert np.isclose(1.0, f)


def test_bell_circuit_with_wrapper_op_1():
    _, ideal_state = bell_state_circuit()
    circuit = CircuitDAG(n_emitter=2, n_classical=0)
    composite_op = ops.OneQubitGateWrapper(
        [ops.Identity, ops.Hadamard], register=0, reg_type="e"
    )
    circuit.add(composite_op)
    circuit.add(ops.CNOT(control=0, control_type="e", target=1, target_type="e"))

    compiler = DensityMatrixCompiler()
    state = compiler.compile(circuit)
    f = fidelity(state.dm.data, ideal_state.dm.data)

    assert np.isclose(1.0, f)


def test_bell_circuit_with_wrapper_op_2():
    _, ideal_state = bell_state_circuit()
    circuit = CircuitDAG(n_emitter=2, n_classical=0)
    composite_op = ops.OneQubitGateWrapper(
        [ops.Phase, ops.Phase, ops.Phase, ops.Phase, ops.Hadamard],
        register=0,
        reg_type="e",
    )
    circuit.add(composite_op)
    circuit.add(ops.CNOT(control=0, control_type="e", target=1, target_type="e"))

    compiler = DensityMatrixCompiler()
    state = compiler.compile(circuit)
    f = fidelity(state.dm.data, ideal_state.dm.data)

    assert np.isclose(1.0, f)


@visualization
def test_bell_circuit_visualization():
    circuit, ideal_state = bell_state_circuit()

    compiler = DensityMatrixCompiler()
    state = compiler.compile(circuit)

    fig, ax = density_matrix_bars(state.dm.data)
    fig.suptitle("Simulated circuit density matrix")
    plt.show()

    fig, ax = density_matrix_bars(ideal_state.dm.data)
    fig.suptitle("Ideal density matrix")
    plt.show()


def test_ghz3_circuit():
    circuit, ideal_state = ghz3_state_circuit()

    compiler = DensityMatrixCompiler()
    state = compiler.compile(circuit)

    state = partial_trace(
        state.dm.data, keep=(0, 1, 2), dims=4 * [2]
    )  # trace out the ancilla qubit

    f = fidelity(state, ideal_state.dm.data)

    assert np.isclose(1.0, f)


@visualization
def test_ghz3_circuit_visualization():
    circuit, ideal_state = ghz3_state_circuit()

    compiler = DensityMatrixCompiler()
    state = compiler.compile(circuit)

    state = partial_trace(
        state.dm.data, keep=(0, 1, 2), dims=4 * [2]
    )  # trace out the ancilla qubit

    fig, ax = density_matrix_bars(state)
    fig.suptitle("Simulated circuit density matrix")
    plt.show()

    fig, ax = density_matrix_bars(ideal_state.dm.data)
    fig.suptitle("Ideal density matrix")
    plt.show()


def test_ghz4_circuit():
    circuit, ideal_state = ghz4_state_circuit()
    # circuit.show()

    compiler = DensityMatrixCompiler()
    state = compiler.compile(circuit)

    state = partial_trace(
        state.dm.data, keep=(0, 1, 2, 3), dims=5 * [2]
    )  # trace out the ancilla qubit

    f = fidelity(state, ideal_state.dm.data)

    assert np.isclose(1.0, f)


@visualization
def test_ghz4_circuit_visualization():
    circuit, ideal_state = ghz4_state_circuit()

    compiler = DensityMatrixCompiler()
    state = compiler.compile(circuit)

    state = partial_trace(
        state.dm.data, keep=(0, 1, 2, 3), dims=5 * [2]
    )  # trace out the ancilla qubit

    fig, ax = density_matrix_bars(state)
    fig.suptitle("Simulated circuit density matrix")
    plt.show()

    fig, ax = density_matrix_bars(ideal_state.dm.data)
    fig.suptitle("Ideal density matrix")
    plt.show()


# add tests for the density matrix_heatmap function
@visualization
def test_density_matrix_heatmap():
    circuit, ideal_state = ghz4_state_circuit()

    compiler = DensityMatrixCompiler()
    state = compiler.compile(circuit)

    state = partial_trace(state.dm.data, keep=(0, 1, 2, 3), dims=5 * [2])

    fig, ax = density_matrix_heatmap(state)
    fig.suptitle("Simulated circuit density matrix")
    plt.show()

    fig, ax = density_matrix_heatmap(ideal_state.dm.data)
    fig.suptitle("Ideal density matrix")
    plt.show()


if __name__ == "__main__":
    test_bell_circuit()
    test_ghz3_circuit()
    test_ghz4_circuit()
