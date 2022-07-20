import qiskit.transpiler.passes
from qiskit import QuantumCircuit
from qiskit.visualization.qcstyle import DefaultStyle
import matplotlib.pyplot as plt


def draw_openqasm(qasm, show=False, ax=None, display_text=None):
    """
    Draw a circuit diagram from an openqasm 2 script

    :param qasm: string corresponding to the openqasm circuit description
    :type qasm: str
    :param show: If True, draw and show the circuit. Otherwise, draw but do not show the circuit
    :type show: bool
    :param ax: axis on which to plot the figure drawing
    :type ax: matplotlib.axis
    :param display_text: a dictionary specifying the symbol to be used for each gate (as defined in openQASM)
    :type display_text: dictionary (or None)
    :return: fig, ax (fig might be None)
    :rtype: matplotlib.figure, matplotlib.axis

    # https://qiskit.org/documentation/tutorials/circuits_advanced/03_advanced_circuit_visualization.html
    # TODO: double check that qc.draw returns a figure
    """
    # display_text_full = DefaultStyle().style['disptex']
    # for key, val in display_text.items():
    #     display_text_full[key] = val
    if display_text is None:
        display_text = {}

    style = {
        'displaytext': display_text
    }

    qc = QuantumCircuit.from_qasm_str(qasm)
    if ax is None:
        fig, ax = plt.subplots()
        qc.draw(output="mpl", ax=ax, plot_barriers=False, style=style)
    else:
        fig = qc.draw(output="mpl", ax=ax, plot_barriers=False, style=style)
    if show:
        plt.show()

    return fig, ax
