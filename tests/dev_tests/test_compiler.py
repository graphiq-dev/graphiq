import numpy as np

from src.circuit import CircuitDAG
from src.ops import *
from src.backends.compiler import DensityMatrixCompiler
from src.backends.state_representations import DensityMatrix


if __name__ == "__main__":
    # init = np.outer(np.array([1, 0]), np.array([1, 0])).astype('complex64')  # initialization of quantum registers
    # dm = DensityMatrix(state_data=init)

    circuit = CircuitDAG(n_quantum=2, n_classical=0)
    circuit.add(Hadamard(register=0))
    circuit.add(CNOT(control=0, target=1))
    circuit.show()

    compiler = DensityMatrixCompiler()
    state = compiler.compile(circuit)

    print(state)
