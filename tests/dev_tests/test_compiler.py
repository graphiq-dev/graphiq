
from src.circuit import CircuitDAG
from src.ops import *
from src.backends.compiler import CompilerBase, DensityMatrixCompiler

if __name__ == "__main__":
    circuit = CircuitDAG(3, 3)
    circuit.add_op(Input(qudits=(0,)))
    circuit.add_op(Input(qudits=(1,)))
    circuit.add_op(CNOT(target=0, control=1))
    circuit.add_op(OperationBase(qudits=(0,), cbits=(0,)))

    

    # compiler = DensityMatrixCompiler()
    # compiler.compile(circuit)
