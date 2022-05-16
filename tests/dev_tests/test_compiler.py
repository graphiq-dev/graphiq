
from src.circuit import CircuitDAG
from src.ops import *
from src.backends.compiler import DensityMatrixCompiler


if __name__ == "__main__":
    circuit = CircuitDAG(2, 0)
    # circuit.add(Input(0))
    circuit.add(Hadamard(register=0))
    circuit.add(CNOT(control=0, target=1))
    # circuit.add(Output(0))
    circuit.show()

    compiler = DensityMatrixCompiler()
    compiler.compile(circuit)
