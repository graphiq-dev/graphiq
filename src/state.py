"""
The QuantumState/GraphState classes mediates the interactions between different graph representations

State representations that we support are:
1. Density matrix
2. Stabilizer

State representations that we intend to support in the near future are:
1. Graph representation

TODO: once we can convert more easily between different representations,
we should REMOVE the requirement that data be a list the same length as the
number of requested representations, and ideally just use a single data
object to initialize all representations
"""
import warnings
import networkx as nx

from src.backends.graph.state import Graph
from src.backends.density_matrix.state import DensityMatrix
from src.backends.stabilizer.state import Stabilizer
import src.backends.density_matrix.functions as dmf
import src.backends.stabilizer.functions.clifford as sfc

# threshold above which density matrix representation is discouraged
DENSITY_MATRIX_QUBIT_THRESH = 10


class QuantumState:
    """
    The QuantumState class is a wrapper class which contains all state objects. This includes
    any state representation which we want active

    It also should also be able to add (where possible) representation which were not present at
    initialization.

    TODO: add a handle to delete specific representations (may be useful to clear out memory)
    """

    def __init__(self, n_qubits, data, representation=None):
        """
        Creates the QuantumState class with certain initial representations

        :param n_qubits: number of qubits in the system (system size)
        :type n_qubits: int
        :param data: valid data input for "representation". If representations are given as a list,
                     the data must be a list of the same length
                     Density matrices representations support np.dnarray or int inputs
                     Stabilizer representations take int or StabilizerTableaus
                     Graph representations take frozenset or int or networkx.Graph or iterable of data pairs
        :type data: list OR numpy.ndarray OR Graph OR nx.Graph
        :param representation: selected representations to initialize
        :type representation: str OR list of str
        :return: function returns nothing
        :rtype: None
        """
        self.n_qubits = n_qubits

        self._dm = None
        self._graph = None
        self._stabilizer = None

        if representation is None:
            if n_qubits < DENSITY_MATRIX_QUBIT_THRESH and DensityMatrix.valid_datatype(
                data
            ):
                self._initialize_dm(data)
            elif Stabilizer.valid_datatype(data):
                self._initialize_stabilizer(data)
            elif Graph.valid_datatype(data):
                self._initialize_graph(data)
            elif DensityMatrix.valid_datatype(data):
                raise ValueError(
                    f"Data's type is correct for density matrix representation, but state size exceeds the "
                    f"recommended size for density matrix representation"
                )
            else:
                raise ValueError(
                    f"Data's type is invalid for initializing a QuantumState"
                )

        elif isinstance(representation, str):
            self._initialize_representation(representation, data)
        elif isinstance(representation, list):
            for rep, dat in zip(representation, data):
                self._initialize_representation(rep, dat)
        else:
            raise ValueError(
                "passed representation argument must be a String or a list of strings"
            )

    def partial_trace(self, keep, dims, measurement_determinism="probabilistic"):
        """
        Calculates the partial trace on all state representations which are currently defined

        :param keep:  An array of indices of the spaces to keep. For instance, if the space is
                    :math:`A \\times B \\times C \\times D` and we want to trace out B and D, keep = [0,2]
        :type keep: list OR numpy.ndarray
        :param dims: An array of the dimensions of each space. For instance,
                    if the space is :math:`A \\times B \\times C \\times D`,
                    dims = [dim_A, dim_B, dim_C, dim_D]
        :type dims: list OR numpy.ndarray
        :param measurement_determinism:
        :type measurement_determinism: str or int
        :return: nothing
        :rtype: None
        """
        if self._dm is not None:
            self.dm.data = dmf.partial_trace(self.dm.data, keep, dims)
        if self._stabilizer is not None:
            self.stabilizer.data = sfc.partial_trace(
                self._stabilizer.data, keep, dims, measurement_determinism
            )
        if self._graph is not None:
            raise NotImplementedError(
                "Partial trace not yet implemented on graph state"
            )

    @property
    def all_representations(self):
        """
        Returns all active representations of a QuantumState object

        :return: a list with all initialized state representations
        :rtype: list
        """
        representations = []
        if self._dm is not None:
            representations.append(self._dm)
        if self._stabilizer is not None:
            representations.append(self._stabilizer)
        if self._graph is not None:
            representations.append(self._graph)

        return representations

    @property
    def dm(self):
        """
        Density matrix representation of our quantum state

        :raises ValueError: if existing representations within the QuantumState object cannot be sent to a
                           density matrix representation AND no density matrix representation is saved
        :return: density matrix representation
        :rtype: DensityMatrix
        """
        if self._dm is not None:
            return self._dm
        # TODO: ATTEMPT TO CONVERT EXISTING REPRESENTATION to dm. This should call on backend functions
        raise ValueError("Cannot convert existing representation to density matrices")

    @dm.setter
    def dm(self, new_dm):
        """
        Allows the density matrix representation of our quantum state to be modified

        :param new_dm: the updated density matrix
        :type new_dm: DensityMatrix
        :return: function returns nothing
        :rtype: None
        """
        if self._dm is None:
            warnings.warn(
                UserWarning(
                    "Density matrix representation being set is not compared to "
                    "previously existing representations. Make sure the new"
                    "representation is consistent with other object representations"
                )
            )
        self._dm = new_dm

    @property
    def graph(self):
        """
        Graph representation of our quantum state

        :raises ValueError: if existing representations within the QuantumState object cannot be sent to a
                           graph representation AND no graph representation is saved
        :return: graph representation
        :rtype: Graph
        """
        if self._graph is not None:
            return self._graph
        # TODO: ATTEMPT TO CONVERT EXISTING REPRESENTATION to graph. This should call on backend functions
        raise ValueError(
            "Cannot convert existing representation to graph representation"
        )

    @graph.setter
    def graph(self, new_graph):
        """
        Allows the graph representation of our quantum state to be modified

        :param new_graph: the updated graph representation
        :type new_graph: Graph
        :return: function returns nothing
        :rtype: None
        """
        if self._graph is None:
            warnings.warn(
                UserWarning(
                    "Graph representation being set is not compared to "
                    "previously existing representations. Make sure the new"
                    "representation is consistent with other object representations"
                )
            )
        self._graph = new_graph

    @property
    def stabilizer(self):
        """
        Stabilizer Formalism representation of our quantum state

        :raises ValueError: if existing representations within the QuantumState object cannot be sent to a
                           stabilizer representation AND no stabilizer representation is saved
        :return: stabilizer representation
        :rtype: Stabilizer
        """
        if self._stabilizer is not None:
            return self._stabilizer
        # TODO: ATTEMPT TO CONVERT EXISTING REPRESENTATION to stabilizer. This should call on backend functions
        raise ValueError(
            "Cannot convert existing representation to stabilizer representation"
        )

    @stabilizer.setter
    def stabilizer(self, new_stabilizer):
        """
        Allows the stabilizer representation of our quantum state to be modified

        :param new_stabilizer: the updated stabilizer representation
        :type new_stabilizer: Stabilizer
        :return: function returns nothing
        :rtype: None
        """
        if self._stabilizer is None:
            warnings.warn(
                UserWarning(
                    "Stabilizer representation being set is not compared to "
                    "previously existing representations. Make sure the new"
                    "representation is consistent with other object representations"
                )
            )
        self._stabilizer = new_stabilizer

    def show(self, representation="all", show=True, ax=None):
        """
        Plots the selected representations (all by default) using matplotlib formatting

        :param representation: 'all' to show all possible representations,
                               list of representation strings otherwise to show specific representations
        :type representation: str OR list (of strs)
        :param show: if True, the selected representations are plotted. Otherwise, they are drawn but not plotted
        :type show: bool
        :param ax: axis/axes on which to plot the selected representations
        :type ax: matplotlib.axis
        :return: fig, ax (the figure and axes on which data was plotted)
        :rtype: matplotlib.figure, matplotlib.axis
        """
        raise NotImplementedError()

    def _initialize_dm(self, data):
        """
        Initializes a density matrix based on the data

        :param data: either a graph or ndarray matrix
        :type data: Graph OR nx.Graph OR numpy.ndarray
        :raises AssertionError: if the density matrix being initialized does not have self.n_qubit
        :return: function returns nothing
        :rtype: None
        """
        if isinstance(data, Graph) or isinstance(data, nx.Graph):
            self._dm = DensityMatrix.from_graph(data)
        else:
            self._dm = DensityMatrix(data)

        assert self._dm.data.shape[0] == self._dm.data.shape[1] == 2**self.n_qubits

    def _initialize_graph(self, data):
        """
        Initializes a graph state based on the data

        :param data: data to construct the Graph representation
        :type data: nx.Graph OR int OR frozenset
        :raises AssertionError: if the graph being initialized does not have self.n_qubit
        :return: function returns nothing
        :rtype: None
        """
        self._graph = Graph(data, 1)
        # TODO: adjust root_node_id field once we've figured out how we want to use it
        assert self._graph.n_qubit == self.n_qubits, (
            f"Expected {self.n_qubits} qubits, "
            f"graph representation has {self._graph.n_qubit}"
        )

    def _initialize_stabilizer(self, data):
        """
        Initializes a stabilizer state based on the data

        :param data: data to construct the stabilizer state representation
        :type data: int or CliffordTableau
        :return: nothing
        :rtype: None
        """
        self._stabilizer = Stabilizer(data)
        assert self._stabilizer.n_qubit == self.n_qubits, (
            f"Expected {self.n_qubits} qubits, "
            f"Stabilizer representation has {self._stabilizer.n_qubit}"
        )

    def _initialize_representation(self, representation, data):
        """
        Helper function to initialize any given representation

        :param representation: representation to initialize
        :type representation: str
        :param data: data with which the representation should be initialized
        :type data: int OR frozenset OR Graph OR nx.Graph OR numpy.ndarray
        :raises ValueError: if representation is invalid
        :return: function returns nothing
        :rtype: None
        """
        if representation == "density matrix":
            if self.n_qubits > DENSITY_MATRIX_QUBIT_THRESH:
                warnings.warn(
                    UserWarning(
                        "Density matrix is not recommended for a state of this size"
                    )
                )
            self._initialize_dm(data)
        elif representation == "graph":
            self._initialize_graph(data)
        elif representation == "stabilizer":
            self._initialize_stabilizer(data)
        else:
            raise ValueError("Passed representation is invalid")
