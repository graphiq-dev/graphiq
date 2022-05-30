from qiskit import QuantumCircuit
import matplotlib.pyplot as plt


def draw_openqasm(qasm, show=False, fig=None, ax=None):
    qc = QuantumCircuit.from_qasm_str(qasm)
    if ax is None:
        fig, ax = plt.subplots()
    qc.draw(output='mpl', ax=ax)
    if show:
        plt.show()

    return fig, ax
