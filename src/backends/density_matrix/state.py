"""
Density Matrix representation for states
"""
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

from src.backends.density_matrix.functions import is_psd, create_n_plus_state, apply_cz
from src.backends.state_base import StateRepresentationBase
from src.backends.graph.state import Graph
from src.visualizers.density_matrix import density_matrix_heatmap, density_matrix_bars


# TODO: accept single input (# of qubits) as input and initialize as unentangled qubits


class DensityMatrix(StateRepresentationBase):
    """
    Density matrix of a graph state
    """

    def __init__(self, data, *args, **kwargs):
        """
        Construct a DensityMatrix object and calculate the density matrix from state_data

        :param data: density matrix or a networkx graph
        :type data: numpy.ndarray
        :return: nothing
        :rtype: None
        """
        super().__init__(data, *args, **kwargs)

        if not isinstance(data, np.ndarray):
            raise TypeError("Input must be a np ndarray")
        else:
            # check if state_data is positive semi-definite
            if not is_psd(data):
                raise ValueError("The input matrix is not a valid density matrix")

            if not np.equal(np.trace(data), 1):
                data = data / np.trace(data)

            self.data = data

    @classmethod
    def from_graph(cls, graph):
        """
        Builds a density matrix representation from a graph (either nx.graph or a Graph representation)

        :param graph: the graph from which we will build a density matrix
        :type graph: networkx.Graph OR Graph
        :raises TypeError: if the input graph is neither nx.graph or Graph
        :return: a DensityMatrix representation with the data contained by graph
        :rtype: DensityMatrix
        """
        # TODO: port this implementation into a conversion-specific python document
        if isinstance(graph, nx.Graph):
            graph_data = graph
        elif isinstance(graph, Graph):
            graph_data = graph.data
        else:
            raise TypeError("Input state must be GraphState object or NetworkX graph.")

        number_qubits = graph_data.number_of_nodes()
        mapping = dict(zip(graph_data.nodes(), range(0, number_qubits)))
        edge_list = list(graph_data.edges)
        final_state = create_n_plus_state(number_qubits)

        for edge in edge_list:
            final_state = apply_cz(final_state, mapping[edge[0]], mapping[edge[1]])
        data = final_state

        return cls(data)

    def apply_unitary(self, unitary):
        """
        Apply a unitary on the state.
        Assumes the dimensions match; Otherwise, raise ValueError

        :param unitary: unitary matrix to apply
        :type unitary: numpy.ndarray
        :raises ValueError: if the density matrix of the state has a different size from the unitary gate to be applied
        :return: function returns nothing
        :rtype: None
        """
        if self._data.shape == unitary.shape:
            self._data = unitary @ self._data @ np.transpose(np.conjugate(unitary))
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
        :return: function returns nothing
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
            self._data = tmp_state
        else:
            raise ValueError("Kraus operators have wrong dimensions.")

    def apply_measurement_TO_DEBUG(
        self, projectors, measurement_determinism="probabilistic"
    ):
        """
        # TODO: replace the other apply_measurement function by this one, and also debug to find out why it
        # gives different tests results in test_benchmark_fidelity.py
        Apply a measurement, either deterministically (with a certain outcome) or probabilistically
        :param projectors: the project which is the measurement to apply
        :type projectors: numpy.ndarray
        :param measurement_determinism: if "probabilistic", measurement results are probabilistically selected
                                    if 1, measurement results default to 1 unless the probability of measuring p(1) = 0
                                    if 0, measurement results default to 0 unless the probability of measuring p(0) = 0
        :rtype _measurement_determinism: str/int
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

            if measurement_determinism == "probabilistic":
                outcome = np.random.choice([0, 1], p=probs / np.sum(probs))
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

            # TODO: this is the dm CONDITIONED on the measurement outcome
            # this assumes that the projector, m, has the properties: m = sqrt(m) and m = m.dag()
            self._data = (m @ self._data @ np.transpose(np.conjugate(m))) / norm

            # TODO: this is the dm *unconditioned* on the outcome
            # self._data = sum([m @ self._data @ m for m in projectors])
        else:
            raise ValueError(
                "The density matrix of the state has a different size from the POVM elements."
            )

    def apply_measurement(self, projectors, measurement_determinism):
        """
        Apply a measurement, either deterministically (with a certain outcome) or probabilistically
        TODO: refactor apply_probabilistic_measurement, apply_deterministic_measurement functions to share code
        :param projectors: the project which is the measurement to apply
        :type projectors: numpy.ndarray
        :param measurement_determinism: if "probabilistic", measurement results are probabilistically selected
                                    if 1, measurement results default to 1 unless the probability of measuring p(1) = 0
                                    if 0, measurement results default to 0 unless the probability of measuring p(0) = 0
        :rtype _measurement_determinism: str/int
        :return: the measurement outcome
        :rtype: int
        """
        if measurement_determinism == "probabilistic":
            return self.apply_probabilistic_measurement(projectors)
        elif measurement_determinism == 1:
            return self.apply_deterministic_measurement(projectors, set_measurement=1)
        elif measurement_determinism == 0:
            return self.apply_deterministic_measurement(projectors, set_measurement=0)

    def apply_probabilistic_measurement(self, projectors):
        """
        Apply the projectors measurement onto the density matrix representation of the state

        :param projectors: the projector which is the measurement to apply
        :type projectors: numpy.ndarray
        :raises ValueError: if the density matrix of the state has a different size from the POVM elements.
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

            outcome = np.random.choice([0, 1], p=probs / np.sum(probs))
            m, norm = projectors[outcome], probs[outcome]
            # TODO: this is the dm CONDITIONED on the measurement outcome
            # this assumes that the projector, m, has the properties: m = sqrt(m) and m = m.dag()
            self._data = (m @ self._data @ np.transpose(np.conjugate(m))) / norm
            # TODO: this is the dm *unconditioned* on the outcome
            # self._data = sum([m @ self._data @ m for m in projectors])
        else:
            raise ValueError(
                "The density matrix of the state has a different size from the POVM elements."
            )

        return outcome

    def apply_deterministic_measurement(self, projectors, set_measurement=None):
        """
        Apply the projectors measurement onto the density matrix representation of the state

        :param projectors: the projector which is the measurement to apply
        :type projectors: numpy.ndarray
        :raises ValueError: if the density matrix of the state has a different size from the POVM elements.
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
            if set_measurement is None or set_measurement == 1:  # default
                # TODO: might be worth it making this "np.isclose"
                if probs[1] > 0:
                    outcome = 1
                else:
                    outcome = 0
            else:
                if probs[1] < 1:
                    outcome = 0
                else:
                    outcome = 1

            m, norm = projectors[outcome], probs[outcome]
            # TODO: this is the dm CONDITIONED on the measurement outcome
            # this assumes that the projector, m, has the properties: m = sqrt(m) and m = m.dag()
            self._data = (m @ self._data @ np.transpose(np.conjugate(m))) / norm
            # TODO: this is the dm *unconditioned* on the outcome
            # self._data = sum([m @ self._data @ m for m in projectors])
        else:
            raise ValueError(
                "The density matrix of the state has a different size from the POVM elements."
            )

        return outcome

    def draw(self, style="bar", show=True):
        """
        Draw a bar graph or heatmap of the DensityMatrix representation data

        :param style: 'bar' for barplot, 'heat' for heatmap
        :type style: str
        :param show: if True, show the density matrix plot. Otherwise, draw the density matrix plot but do not show
        :type show: bool
        :return: fig, axes on which the state is drawn
        :rtype: matplotlib.figure, matplotlib.axes

        TODO: add a "ax" parameter to match the other viewing utils
        """
        if style == "bar":
            fig, axs = density_matrix_bars(self.data)
        else:
            fig, axs = density_matrix_heatmap(self.data)

        if show:
            plt.show()

        return fig, axs
