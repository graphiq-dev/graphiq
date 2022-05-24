import networkx as nx
import numpy as np

from src.backends.density_matrix.functions import is_psd, create_n_plus_state, apply_CZ
from src.states import StateRepresentationBase, GraphState


class DensityMatrix(StateRepresentationBase):
    """
    Density matrix of a graph state
    """
    def __init__(self, data, *args, **kwargs):
        """
        Construct a DensityMatrix object and calculate the density matrix from state_data
        :param data: density matrix or a networkx graph
        """
        super().__init__(data, *args, **kwargs)

        if not isinstance(data, np.ndarray):
            raise TypeError("Input must be a np array")

        else:
            # check if state_data is positive semi-definite
            if not is_psd(data):
                raise ValueError('The input matrix is not a valid density matrix')
            if not np.equal(np.trace(data), 1):
                data = data / np.trace(data)

            self.data = data

    @classmethod
    def from_graph(cls, graph):
        if isinstance(graph, nx.Graph):
            graph_data = graph
        elif isinstance(graph, GraphState):
            graph_data = graph.data()
        else:
            raise TypeError("Input state must be GraphState object or NetworkX graph.")

        number_qubits = graph_data.number_of_nodes()
        mapping = dict(zip(graph_data.nodes(), range(0, number_qubits)))
        edge_list = list(graph_data.edges)
        final_state = create_n_plus_state(number_qubits)

        for edge in edge_list:
            final_state = apply_CZ(final_state, mapping[edge[0]], mapping[edge[1]])
        data = final_state

        return cls(data)

    def apply_unitary(self, unitary):
        """
        Apply a unitary on the state.
        Assuming the dimensions match; Otherwise, raise ValueError
        """
        if self._data.shape == unitary.shape:
            self._data = unitary @ self._data @ np.transpose(np.conjugate(unitary))
        else:
            raise ValueError('The density matrix of the state has a different size from the unitary gate to be applied.')

    def apply_measurement(self, projectors):
        if self._data.shape == projectors[0].shape:
            probs = [np.real(np.trace(self._data @ m)) for m in projectors]

            outcome = np.random.choice([0, 1], p=probs/np.sum(probs))
            m, norm = projectors[outcome], probs[outcome]
            # TODO: this is the dm CONDITIONED on the measurement outcome
            # this assumes that the projector, m, has the properties: m = sqrt(m) and m = m.dag()
            self._data = (m @ self._data @ np.transpose(np.conjugate(m))) / norm

            # TODO: this is the dm *unconditioned* on the outcome
            # self._data = sum([m @ self._data @ m for m in projectors])
        else:
            raise ValueError('The density matrix of the state has a different size from the POVM elements.')
        return outcome