from src.visualizers.openqasm_visualization import draw_openqasm
from src.circuit import CircuitDAG
import matplotlib.pyplot as plt
import src.ops as ops

# Empty openQASM
dag = CircuitDAG()
dag.draw_circuit()

# no operations, but some registers
dag.add_quantum_register(size=3)
dag.add_quantum_register(size=3)
dag.add_classical_register(size=1)
dag.draw_circuit()

# Add CNOT operations
dag.add(ops.CNOT(control=1, target=0))
dag.validate()
dag.draw_circuit()

# Add unitary gates
dag.add(ops.SigmaX(register=(0, 0)))
dag.add(ops.Hadamard(register=1))
dag.draw_circuit()

# Create a dag with every gate once
full_dag = CircuitDAG()
full_dag.add_quantum_register(1)
full_dag.add_quantum_register(1)
full_dag.add_classical_register(1)

full_dag.add(ops.Hadamard(register=0))
full_dag.add(ops.SigmaX(register=0))
full_dag.add(ops.SigmaY(register=0))
full_dag.add(ops.SigmaZ(register=0))
full_dag.add(ops.CNOT(control=0, target=1))
full_dag.add(ops.CPhase(control=1, target=0))
full_dag.add(ops.MeasurementZ(register=0, c_register=0))
fig, ax = dag.draw_circuit(show=False)
fig.suptitle("testing fig reception")
plt.show()
