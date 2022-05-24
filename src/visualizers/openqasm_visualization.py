from qiskit import QuantumCircuit


def draw_openqasm(qasm):
    qc = QuantumCircuit.from_qasm_str(qasm)
    qc.draw(output='mpl')