from qiskit import QuantumCircuit
import matplotlib.pyplot as plt


def draw_openqasm(qasm, show=False, ax=None):
    qc = QuantumCircuit.from_qasm_str(qasm)
    qc.draw(output='mpl', ax=ax)
    if show:
        plt.show()
