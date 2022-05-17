"""
Examples of defining and simulating quantum circuits for a variety of small quantum states
"""
import numpy as np

from src.circuit import CircuitDAG
from src.ops import *
from src.backends.compiler import DensityMatrixCompiler

from src.utils.matrix_funcs import tensor, partial_trace

if __name__ == "__main__":
    """
    Two qubit Bell state
    """
    # b0 = np.array([[1, 0]])
    # b1 = np.array([[0, 1]])
    #
    # bell_ideal = 0.5 * (
    #         np.outer(tensor(2 * [b0]), tensor(2 * [b0]))
    #         + np.outer(tensor(2 * [b0]), tensor(2 * [b1]))
    #         + np.outer(tensor(2 * [b1]), tensor(2 * [b0]))
    #         + np.outer(tensor(2 * [b1]), tensor(2 * [b1]))
    # )
    #
    # print(bell_ideal)
    #
    # circuit = CircuitDAG(2, 0)
    # circuit.add(Hadamard(register=0))
    # circuit.add(CNOT(control=0, target=1))
    # circuit.show()
    #
    # compiler = DensityMatrixCompiler()
    # state = compiler.compile(circuit)
    # print(state)

    """
    Three qubit GHZ state
    (not complete)
    """
    b0 = np.array([[1, 0]])
    b1 = np.array([[0, 1]])

    rhoGHZ3_ideal = 0.5 * (
        np.outer(tensor(3*[b0]), tensor(3*[b0]))
        + np.outer(tensor(3*[b0]), tensor(3*[b1]))
        + np.outer(tensor(3*[b1]), tensor(3*[b0]))
        + np.outer(tensor(3*[b1]), tensor(3*[b1]))
    )

    print(rhoGHZ3_ideal)

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

    # state = np.identity(2**4)

    state_partial = partial_trace(state, keep=[0, 1, 2], dims=4*[2])
    print(state_partial)
    print(state_partial.shape)