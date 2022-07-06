import qiskit.transpiler.passes
from qiskit import QuantumCircuit
import matplotlib.pyplot as plt


def draw_openqasm(qasm, show=False, ax=None):
    """
    Draw a circuit diagram from an openqasm 2 script

    :param qasm: string corresponding to the openqasm circuit description
    :type qasm: str
    :param show: If True, draw and show the circuit. Otherwise, draw but do not show the circuit
    :type show: bool
    :param ax: axis on which to plot the figure drawing
    :type ax: matplotlib.axis
    :return: fig, ax (fig might be None)
    :rtype: matplotlib.figure, matplotlib.axis

    # TODO: double check that qc.draw returns a figure
    """
    qc = QuantumCircuit.from_qasm_str(qasm)
    if ax is None:
        fig, ax = plt.subplots()
        qc.draw(output="mpl", ax=ax, plot_barriers=False)
    else:
        fig = qc.draw(output="mpl", ax=ax, plot_barriers=False)
    if show:
        plt.show()

    return fig, ax
