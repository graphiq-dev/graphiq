import networkx as nx

from src.circuit import CircuitDAG
from src.ops import OperationBase, CNOT


# Test initialization
circuit1 = CircuitDAG(2, 0)
# circuit1.show()
circuit1.validate()

circuit2 = CircuitDAG(3, 2)
# circuit2.show()
circuit2.validate()

# Test add comp
circuit3 = CircuitDAG(2, 0)
circuit3.add(OperationBase(q_registers=(0,)))
circuit3.validate()
# circuit3.show()
circuit3.add(OperationBase(q_registers=(0, 1)))
circuit3.validate()
# circuit3.show()


# Test add comp: qiskit example https://qiskit.org/documentation/stubs/qiskit.converters.circuit_to_dag.html
circuit4 = CircuitDAG(3, 3)
circuit4.add(OperationBase(q_registers=(0,)))
circuit4.add(OperationBase(q_registers=(0, 1)))
circuit4.add(OperationBase(q_registers=(0,), c_registers=(0,)))
circuit4.add(OperationBase(q_registers=(1,), c_registers=(0, 1, 2)))
# circuit4.show()
circuit4.validate()

# test topological order operation list
print(circuit1.sequence())
print(circuit2.sequence())
print(circuit3.sequence())
print(circuit4.sequence())

# test dynamic dealing with register number (copied from test_circuit)
dag = CircuitDAG(1, 0)
op1 = OperationBase(q_registers=(1, 2))
op2 = OperationBase(q_registers=(2,))
op3 = OperationBase(q_registers=(0,), c_registers=(1, 0))
dag.add(op1)
dag.add(op2)
dag.add(op3)
dag.validate()

# dag.show()

# test initializing registers (copied from test_circuit)
dag = CircuitDAG(2, 2)
dag.add_quantum_register()
dag.add(OperationBase(q_registers=(0,)))
dag.add(OperationBase(q_registers=(1, 0)))
dag.add_classical_register(2)
# dag.show()

# Test registers
dag = CircuitDAG(2, 0)
dag.expand_quantum_register(0, 2)
dag.expand_quantum_register(1, 2)
dag.add(CNOT(control=0, target=1))
dag.validate()
# dag.show()
for edge in dag.dag.edges:
    print(edge)
    print(dag.dag.edges[edge])