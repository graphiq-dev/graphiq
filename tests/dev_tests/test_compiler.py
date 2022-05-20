import pytest

from scipy.linalg import sqrtm
import numpy as np
import matplotlib.pyplot as plt

from src.circuit import CircuitDAG
from src.ops import *
from src.backends.density_matrix.compiler import DensityMatrixCompiler
from src.backends.density_matrix.functions import partial_trace
from src.backends.state_representations import DensityMatrix
from src.libraries.circuits import ghz3_state_circuit, bell_state_circuit, ghz4_state_circuit
from src.visualizers.density_matrix import density_matrix_bars

plot = True


def fidelity(rho, sigma):
    return np.trace(sqrtm(sqrtm(rho) @ sigma @ sqrtm(rho))) ** 2


def test_bell_circuit():
    circuit, ideal_state = bell_state_circuit()

    compiler = DensityMatrixCompiler()
    state = compiler.compile(circuit)

    state = state.rep

    f = fidelity(state, ideal_state['dm'])

    if plot:
        print(f"Fidelity is {f}")
        circuit.show()

        fig, ax = density_matrix_bars(state)
        fig.suptitle("Simulated circuit density matrix")
        plt.show()

        fig, ax = density_matrix_bars(ideal_state['dm'])
        fig.suptitle("Ideal density matrix")
        plt.show()

    assert np.isclose(1.0, f)


def test_ghz3_circuit():
    circuit, ideal_state = ghz3_state_circuit()

    compiler = DensityMatrixCompiler()
    state = compiler.compile(circuit)

    state = partial_trace(state.rep, keep=(0, 1, 2), dims=4*[2])  # trace out the ancilla qubit

    f = fidelity(state, ideal_state['dm'])

    if plot:
        print(f"Fidelity is {f}")
        circuit.show()

        fig, ax = density_matrix_bars(state)
        fig.suptitle("Simulated circuit density matrix")
        plt.show()

        fig, ax = density_matrix_bars(ideal_state['dm'])
        fig.suptitle("Ideal density matrix")
        plt.show()

    assert np.isclose(1.0, f)


def test_ghz4_circuit():
    circuit, ideal_state = ghz4_state_circuit()
    # circuit.show()

    compiler = DensityMatrixCompiler()
    state = compiler.compile(circuit)

    state = partial_trace(state.rep, keep=(0, 1, 2, 3), dims=5*[2])  # trace out the ancilla qubit

    f = fidelity(state, ideal_state['dm'])

    if plot:
        print(f"Fidelity is {f}")
        circuit.show()

        fig, ax = density_matrix_bars(state)
        fig.suptitle("Simulated circuit density matrix")
        plt.show()

        fig, ax = density_matrix_bars(ideal_state['dm'])
        fig.suptitle("Ideal density matrix")
        plt.show()
        
    assert np.isclose(1.0, f)


if __name__ == "__main__":
    test_bell_circuit()
    test_ghz3_circuit()
    test_ghz4_circuit()
