# Copyright (c) 2022-2024 Quantum Bridge Technologies Inc.
# Copyright (c) 2022-2024 Ki3 Photonics Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit


def draw_openqasm(qasm, show=False, ax=None, display_text=None):
    """
    Draw a circuit diagram from an openqasm 2 script

    :param qasm: string corresponding to the openqasm circuit description
    :type qasm: str
    :param show: If True, draw and show the circuit. Otherwise, draw but do not show the circuit
    :type show: bool
    :param ax: axis on which to plot the figure drawing
    :type ax: matplotlib.Axis
    :param display_text: a dictionary specifying the symbol to be used for each gate (as defined in openQASM)
    :type display_text: dictionary (or None)
    :return: fig, ax (fig might be None)
    :rtype: matplotlib.Figure, matplotlib.Axis

    """
    # https://qiskit.org/documentation/tutorials/circuits_advanced/03_advanced_circuit_visualization.html
    # TODO: double check that qc.draw returns a figure
    # TODO: implement a way to display the parameters in brackets: Rz(np.pi/2)
    # display_text_full = DefaultStyle().style['disptex']
    # for key, val in display_text.items():
    #     display_text_full[key] = val
    if display_text is None:
        display_text = {}

    style = {"displaytext": display_text}

    qc = QuantumCircuit.from_qasm_str(qasm)
    if ax is None:
        fig, ax = plt.subplots()
        qc.draw(output="mpl", ax=ax, plot_barriers=False, style=style)
    else:
        fig = qc.draw(output="mpl", ax=ax, plot_barriers=False, style=style)
    if show:
        # if the num_gates is small increase font size, if large decrease font size
        num_gates = len(qc.data)
        if num_gates < 10:
            for text in ax.texts:
                text.set_fontsize(10)
        elif num_gates < 20:
            for text in ax.texts:
                text.set_fontsize(7)
        else:
            for text in ax.texts:
                text.set_fontsize(8)

        plt.show()

    return fig, ax
