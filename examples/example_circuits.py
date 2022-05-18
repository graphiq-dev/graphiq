"""
Examples of defining and simulating quantum circuits for a variety of small quantum states
"""
import numpy as np

from src.circuit import CircuitDAG
from src.ops import *
from src.backends.compiler import DensityMatrixCompiler

from src.backends.density_matrix_functions import ketz0_state, ketz1_state, tensor, partial_trace, ket2dm

import networkx as nx
import matplotlib.pyplot as plt


def bell_state_circuit():
    """
    Two qubit Bell state preparation circuit
    """
    ideal = ket2dm((tensor(2 * [ketz0_state()]) + tensor(2 * [ketz1_state()])) / np.sqrt(2))

    circuit = CircuitDAG(2, 0)
    circuit.add(Hadamard(register=0))
    circuit.add(CNOT(control=0, target=1))
    circuit.show()

    compiler = DensityMatrixCompiler()
    state = compiler.compile(circuit)
    print(ideal)
    print(state)
    return state, ideal


def ghz3_state_circuit():
    """
    Three qubit GHZ state
    """
    ideal = ket2dm((tensor(3 * [ketz0_state()]) + tensor(3 * [ketz1_state()])) / np.sqrt(3))

    circuit = CircuitDAG(4, 0)
    circuit.add(Hadamard(register=3))
    circuit.add(CNOT(control=3, target=0))
    circuit.add(CNOT(control=3, target=1))
    circuit.add(CNOT(control=3, target=2))
    # circuit.add(Hadamard(register=1))
    # circuit.add(Hadamard(register=2))
    circuit.add(Hadamard(register=3))
    circuit.add(CNOT(control=3, target=2))

    circuit.show()

    compiler = DensityMatrixCompiler()
    state = compiler.compile(circuit)
    partial_trace(state, keep=[0, 1, 2], dims=[2, 2, 2, 2])
    print(ideal)
    print(state)


if __name__ == "__main__":

    state, ideal = bell_state_circuit()
    # pstate = partial_trace(state, keep=[0], dims=[2, 2])
    # print(pstate)
    # ghz3_state_circuit()