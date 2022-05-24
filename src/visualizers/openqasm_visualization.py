from qiskit import QuantumCircuit
import matplotlib.pyplot as plt

def draw_openqasm(qasm, show=False):
    qc = QuantumCircuit.from_qasm_str(qasm)
    qc.draw(output='mpl')
    if show:
        plt.show()