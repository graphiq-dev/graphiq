import networkx as nx

from src.circuit import CircuitDAG
from src.ops import Operation

# Test initialization
circuit1 = CircuitDAG(2, 0)
# circuit1.show()
circuit1.validate()

circuit2 = CircuitDAG(3, 2)
# circuit2.show()
circuit2.validate()

# Test add comp
circuit3 = CircuitDAG(2, 0)
circuit3.add_op(Operation(qudits=(0,)))
circuit3.validate()
# circuit3.show()
circuit3.add_op(Operation(qudits=(0, 1)))
circuit3.validate()
# circuit3.show()

# Test add comp: qiskit example https://qiskit.org/documentation/stubs/qiskit.converters.circuit_to_dag.html
circuit4 = CircuitDAG(3, 3)
circuit4.add_op(Operation(qudits=(0,)))
circuit4.add_op(Operation(qudits=(0, 1)))
circuit4.add_op(Operation(qudits=(0,), cbits=(0,)))
circuit4.add_op(Operation(qudits=(1,), cbits=(0, 1, 2)))
circuit4.show()
circuit4.validate()
