"""
The QuantumState/GraphState classes mediates the interactions between different graph representations

State representations currently implemented are
1. Graph representation
2. Density matrix
3. Stabilizer

TODO: once we can convert more easily between different representations,
we should REMOVE the requirement that data be a list the same length as the
number of requested representations, and ideally just use a single data
object to initialize all representations
"""
import warnings
import networkx as nx

from src.backends.graph.state import Graph
from src.backends.density_matrix.state import DensityMatrix

DENSITY_MATRIX_QUBIT_THRESH = 10


class QuantumState:
    """
    The QuantumState class is a wrapper class which contains all state objects
    """
    def __init__(self, n_qubit, data, representation=None):
        """
        Creates the QuantumState class with certain initial representations

        :param n_qubit: number of qubits in the system (system size)
        :param data: valid data input for "representation". If representations are given as a list,
                     the data must be a list of the same length
        :param representation: string or list of strings selecting the representations to initialize
        """
        self.n_qubit = n_qubit

        self._dm = None
        self._graph = None
        self._stabilizer = None

        if representation is None:
            if n_qubit < DENSITY_MATRIX_QUBIT_THRESH:
                self._initialize_dm(data)
            else:
                # TODO: initialize with stabilizer once the stabilizer is implemented
                self._initialize_graph(data)
        elif isinstance(representation, str):
            self._initialize_representation(representation, data)
        elif isinstance(representation, list):
            for rep, dat in zip(representation, data):
                self._initialize_representation(rep, dat)
        else:
            raise ValueError("passed representation argument must be a String or a list of strings")

    @property
    def dm(self):
        if self._dm is not None:
            return self._dm
        # TODO: ATTEMPT TO CONVERT EXISTING REPRESENTATION to dm
        # This should /call on backend functions/ (the implementation should not be here)
        raise ValueError("Cannot convert existing representation to density matrices")

    @dm.setter
    def dm(self, new_dm):
        if self._dm is None:
            warnings.warn(UserWarning('Density matrix representation being set is not compared to '
                                      'previously existing representations. Make sure the new'
                                      'representation is consistent with other object representations'))
        self._dm = new_dm

    @property
    def graph(self):
        if self._graph is not None:
            return self._graph
        # TODO: ATTEMPT TO CONVERT EXISTING REPRESENTATION to graph
        # This should /call on backend functions/ (the implementation should not be here)
        raise ValueError("Cannot convert existing representation to graph representation")

    @graph.setter
    def graph(self, new_graph):
        if self._graph is None:
            warnings.warn(UserWarning('Graph representation being set is not compared to '
                                      'previously existing representations. Make sure the new'
                                      'representation is consistent with other object representations'))
        self._graph = new_graph

    @property
    def stabilizer(self):
        if self._stabilizer is not None:
            return self._stabilizer
        # TODO: ATTEMPT TO CONVERT EXISTING REPRESENTATION to stabilizer
        # This should /call on backend functions/ (the implementation should not be here)
        raise ValueError("Cannot convert existing representation to stabilizer representation")

    @stabilizer.setter
    def stabilizer(self, new_stabilizer):
        if self._stabilizer is None:
            warnings.warn(UserWarning('Stabilizer representation being set is not compared to '
                                      'previously existing representations. Make sure the new'
                                      'representation is consistent with other object representations'))
        self._stabilizer = new_stabilizer

    def show(self, representation='all'):
        """
        Shows the desired representations (all by default)
        :param representation: representations to show (string or list)
        :return:
        """
        raise NotImplementedError()

    def _initialize_dm(self, data):
        """
        Initializes density matrix based on the data
        :param data: either graph data or ndarray
        """
        if isinstance(data, Graph) or isinstance(data, nx.Graph):
            self._dm = DensityMatrix.from_graph(data)
        else:
            self._dm = DensityMatrix(data)

        assert self._dm.data.shape[0] == self._dm.data.shape[1] == 2**self.n_qubit

    def _initialize_graph(self, data):
        """
        TODO: fill this out
        :param data:
        :return:
        """
        # TODO: figure out how to deal with the root_node_id requirement
        self._graph = Graph(data, 1)
        assert self._graph.n_qubit() == self.n_qubit, f'Expected {self.n_qubit} qubits, graph representation has {self._graph.n_qubit}'

    def _initialize_representation(self, representation, data):
        if representation == 'density matrix':
            if self.n_qubit > DENSITY_MATRIX_QUBIT_THRESH:
                warnings.warn(UserWarning("Density matrix is not recommended for a state of this size"))
            self._initialize_dm(data)
        elif representation == 'graph':
            self._initialize_graph(data)
        elif representation == 'stabilizer':
            raise NotImplementedError("Stabilizer representation not implemented yet")
        else:
            raise ValueError("Passed representation is invalid")


class GraphState(QuantumState):
    """

    """
