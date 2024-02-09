"""
Density matrix representation for states.
Supports unitary operations, quantum channels, and state measurements.
"""
import matplotlib.pyplot as plt
import numpy

import graphiq.backends.density_matrix.functions as dmf
from graphiq.backends.density_matrix import numpy as np
from graphiq.backends.graph.state import Graph
from graphiq.backends.state_base import StateRepresentationBase
from graphiq.backends.state_rep_conversion import graph_to_density
from graphiq.visualizers.density_matrix import (
    density_matrix_heatmap,
    density_matrix_bars,
)


class DensityMatrix(StateRepresentationBase):
    """
    Density matrix of a state
    """

    def __init__(self, data, normalized=True, *args, **kwargs):
        """
        Construct a DensityMatrix object from a numpy.ndarray or from the number of qubits. If an integer is specified,
        then the state is initialized as a product state of :math:`|0\\rangle` with the given number of qubits.

        :param data: density matrix or the number of qubits
        :type data: numpy.ndarray or int
        :param normalized: whether the state is normalized
        :type normalized: bool
        :return: nothing
        :rtype: None
        """

        if isinstance(data, np.ndarray):
            if not dmf.is_psd(data):
                # check if state_data is positive semi-definite
                raise ValueError("The input matrix is not a valid density matrix")

            if normalized and not np.equal(np.trace(data), 1):
                data = data / np.trace(data)
        elif isinstance(data, int):
            # initialize as a tensor product of |0> state
            data = dmf.create_n_product_state(data, dmf.state_ketz0())
        else:
            raise TypeError("Input must be a numpy.ndarray or an integer")

        super().__init__(data, *args, **kwargs)

    @classmethod
    def from_graph(cls, graph):
        """
        Builds a density matrix representation from a graph (either nx.Graph or a Graph representation)

        :param graph: the graph from which we will build a density matrix
        :type graph: networkx.Graph OR Graph
        :raises TypeError: if the input graph is neither nx.Graph or Graph
        :return: a DensityMatrix representation with the data contained by graph
        :rtype: DensityMatrix
        """
        if isinstance(graph, Graph):
            return cls(graph_to_density(graph.data))
        else:
            return cls(graph_to_density(graph))

    @classmethod
    def valid_datatype(cls, data):
        return isinstance(data, (int, np.ndarray))

    @property
    def trace(self):
        """
        Return the trace of the state

        :return: the trace of the state
        :rtype: float
        """
        return np.trace(self.data)

    @property
    def normalized(self):
        """
        Return whether the state is normalized, that is, trace is 1

        :return: whether the state is normalized
        :rtype: bool
        """
        return np.allclose(self.trace, 1.0)

    def apply_unitary(self, unitary):
        """
        Apply a unitary to the state.
        Assumes the dimensions match; Otherwise, raise ValueError

        :param unitary: unitary matrix to apply
        :type unitary: numpy.ndarray
        :raises ValueError: if the density matrix of the state has a different size from the unitary gate to be applied
        :return: nothing
        :rtype: None
        """
        if self._data.shape == unitary.shape:
            self._data = unitary @ self._data @ np.transpose(np.conjugate(unitary))
            # to avoid small numerical error that causes the state non-Hermitian
            self._data = dmf.hermitianize(self._data)
        else:
            raise ValueError(
                "The density matrix of the state has a different size from the unitary gate to be applied."
            )

    def apply_channel(self, kraus_ops):
        """
        Apply a quantum channel on the state where the quantum channel is described by Kraus representation.
        Assumes the dimensions match; Otherwise, raise ValueError

        :param kraus_ops: a list of Kraus operators of the channel
        :type kraus_ops: list[numpy.ndarray]
        :raises ValueError: if Kraus operators have wrong dimensions.
        :return: nothing
        :rtype: None
        """
        tmp_state = 0
        if len(kraus_ops) == 0:
            return
        if self._data.shape[0] == kraus_ops[0].shape[1]:
            for i in range(len(kraus_ops)):
                tmp_state = tmp_state + kraus_ops[i] @ self._data @ np.conjugate(
                    kraus_ops[i].T
                )
            # to avoid small numerical error that causes the state non-Hermitian
            self._data = dmf.hermitianize(tmp_state)
        else:
            raise ValueError("Kraus operators have wrong dimensions.")

    def apply_measurement(self, projectors, measurement_determinism="probabilistic"):
        """
        Apply a measurement, either deterministically (with a certain outcome) or probabilistically

        :param projectors: a list of projective measurements in the computational basis
        :type projectors: list[numpy.ndarray]
        :param measurement_determinism: if "probabilistic", measurement results are probabilistically selected
                                    if 1, measurement results default to 1 unless the probability of measuring p(1) = 0
                                    if 0, measurement results default to 0 unless the probability of measuring p(0) = 0
        :type measurement_determinism: str/int
        :return: the measurement outcome
        :rtype: int
        """

        if self._data.shape == projectors[0].shape:
            probs = []
            for m in projectors:
                prob = np.real(np.trace(self._data @ m))
                if prob < 0:
                    prob = 0
                probs.append(prob)
            probs = np.array(probs)
            if measurement_determinism == "probabilistic":
                outcome = numpy.random.choice([0, 1], p=probs / np.sum(probs))
            elif measurement_determinism == 1:
                if probs[1] > 0:
                    outcome = 1
                else:
                    outcome = 0

            elif measurement_determinism == 0:
                if probs[1] < 1:
                    outcome = 0
                else:
                    outcome = 1
            else:
                raise ValueError(
                    f'measurement_determinism parameter must be "probabilistic", 0, or 1'
                )

            m, norm = projectors[outcome], probs[outcome]

            # this assumes that the projector, m, has the properties: m = sqrt(m) and m = m.dag()
            self._data = (m @ self._data @ np.transpose(np.conjugate(m))) / norm

        else:
            raise ValueError(
                "The density matrix of the state has a different size from the POVM elements."
            )
        return outcome

    def apply_measurement_controlled_gate(
        self, projectors, target_gate, measurement_determinism=1
    ):
        """
        Apply a measurement, either deterministically (with a certain outcome) or probabilistically
        and conditioned on the measurement outcome, apply the target_gate

        :param projectors: a list of projective measurements in the computational basis
        :type projectors: list[numpy.ndarray]
        :param target_gate: the gate to be applied if the measurement outcome is 1
        :type target_gate: numpy.ndarray
        :param measurement_determinism: if "probabilistic", measurement results are probabilistically selected
                                    if 1, measurement results default to 1 unless the probability of measuring p(1) = 0
                                    if 0, measurement results default to 0 unless the probability of measuring p(0) = 0
        :type measurement_determinism: str/int
        :raises AssertionError: if target_gate has different dimensions from the density matrix of the state
        :return: the measurement outcome
        :rtype: int
        """
        assert self._data.shape == target_gate.shape
        outcome = self.apply_measurement(projectors, measurement_determinism)
        if outcome == 1:
            self.apply_unitary(target_gate)
        return outcome

    def partial_trace(self, keep, dims):
        """
        Take the partial trace of the state

        :param keep:  An array of indices of the spaces to keep. For instance, if the space is
                    :math:`A \\times B \\times C \\times D` and we want to trace out B and D, keep = [0,2]
        :type keep: list OR numpy.ndarray
        :param dims: An array of the dimensions of each space. For instance,
                    if the space is :math:`A \\times B \\times C \\times D`,
                    dims = [dim_A, dim_B, dim_C, dim_D]
        :type dims: list OR numpy.ndarray
        :return:
        :rtype:
        """
        self.data = dmf.partial_trace(self.data, keep, dims)

    def draw(self, style="bar", show=True):
        """
        Draw a bar graph or heatmap of the DensityMatrix representation data

        :param style: 'bar' for bar plot, 'heat' for heatmap
        :type style: str
        :param show: if True, show the density matrix plot. Otherwise, draw the density matrix plot but do not show
        :type show: bool
        :return: fig, axes on which the state is drawn
        :rtype: matplotlib.figure, matplotlib.axes

        """
        # TODO: add a "ax" parameter to match the other viewing utils
        if style == "bar":
            fig, axs = density_matrix_bars(self.data)
        else:
            fig, axs = density_matrix_heatmap(self.data)

        if show:
            plt.show()

        return fig, axs

    def __eq__(self, other):
        """
        Compare two DensityMatrix objects and return True if the underlying density matrices are equal
        (up to precision)

        :param other: another DensityMatrix object
        :type other: DensityMatrix
        :return: True if they are equal; False otherwise
        :rtype: bool
        """
        return np.allclose(self._data, other.data)
