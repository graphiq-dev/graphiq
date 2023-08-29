"""
The QuantumState class mediates the interactions between different state representations

State representations that we support are:
    1. Density matrix
    2. Stabilizer

State representations that we intend to support in the near future are:
    1. Graph representation

"""

import warnings
import networkx as nx
import numpy as np

from src.backends.graph.state import Graph, MixedGraph
from src.backends.density_matrix.state import DensityMatrix
from src.backends.stabilizer.state import Stabilizer, MixedStabilizer
from src.backends.stabilizer.clifford_tableau import CliffordTableau
import src.backends.density_matrix.functions as dmf
import src.backends.stabilizer.functions.clifford as sfc
import src.backends.state_rep_conversion as rc

# threshold above which density matrix representation is discouraged
DENSITY_MATRIX_QUBIT_THRESH = 10
# method to handle mixed stabilizer state
STABILIZER_MIXED_REPRESENTATION = "branch"


class QuantumState:
    """
    The QuantumState class is the unified API for accessing state representation backends.
    It contains one state representation. To hold multiple representations, use QuantumStateGadget class
    """

    def __init__(self, data, representation=None, mixed=False):
        """
        Creates the QuantumState class with one initial representation

        :param data: valid data input for "representation".
                     Density matrices representations support np.ndarray or int inputs
                     Stabilizer representations take int or StabilizerTableau
                     Graph representations take frozenset or int or networkx.Graph or iterable of data pairs
        :type data: list OR numpy.ndarray OR Graph OR nx.Graph or CliffordTableau
        :param representation: selected representation to initialize;
                                if not specified, the default choice is the density matrix if the number of qubits is
                                less than the threshold value or stabilizer otherwise.
        :type representation: str
        :param mixed: boolean flag to initialize as a mixed state or not (mainly used for Stabilizer representation)
        :type mixed: boolean
        :return: nothing
        :rtype: None
        """

        self.mixed = mixed
        self._rep_type = representation
        valid, self.n_qubits = self.validate_data(data)

        if not valid:
            raise TypeError(
                f"Data's type is incorrect. Input data is {data}, which is of type {type(data)}"
            )
        if representation is None:
            if (
                self.n_qubits < DENSITY_MATRIX_QUBIT_THRESH
                and DensityMatrix.valid_datatype(data)
            ):
                self._rep_data = self._initialize_dm(data)
            elif Stabilizer.valid_datatype(data):
                self._rep_data = self._initialize_stabilizer(data)
            elif Graph.valid_datatype(data):
                self._rep_data = self._initialize_graph(data)
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
            self._rep_data = self._initialize_representation(representation, data)

        else:
            raise ValueError("passed representation argument must be a String")

    @property
    def rep_data(self):
        return self._rep_data

    @rep_data.setter
    def rep_data(self, data):
        self._rep_data = self._initialize_representation(self._rep_type, data)

    @property
    def rep_type(self):
        return self._rep_type

    def partial_trace(self, keep, dims):
        """
        Calculates the partial trace on all state representations which are currently defined

        :param keep:  An array of indices of the spaces to keep. For instance, if the space is
                    :math:`A \\times B \\times C \\times D` and we want to trace out B and D, keep = [0,2]
        :type keep: list OR numpy.ndarray
        :param dims: An array of the dimensions of each space. For instance,
                    if the space is :math:`A \\times B \\times C \\times D`,
                    dims = [dim_A, dim_B, dim_C, dim_D]
        :type dims: list OR numpy.ndarray
        :return: nothing
        :rtype: None
        """

        if self._rep_type is "graph":
            raise NotImplementedError(
                "Partial trace not yet implemented on graph state"
            )
        else:
            self._rep_data.partial_trace(keep, dims)

    def show(self, show=True, ax=None):
        """
        Plots the selected representations (all by default) using matplotlib formatting

        :param show: if True, the selected representations are plotted. Otherwise, they are drawn but not plotted
        :type show: bool
        :param ax: axis/axes on which to plot the selected representations
        :type ax: matplotlib.axis
        :return: fig, ax (the figure and axes on which data was plotted)
        :rtype: matplotlib.figure, matplotlib.axis
        """
        # TODO: implement this or delete this
        # Use visualizers module
        raise NotImplementedError()

    def _initialize_dm(self, data):
        """
        Initializes a density matrix based on the data

        :param data: either a graph or ndarray matrix
        :type data: Graph OR nx.Graph OR numpy.ndarray
        :raises AssertionError: if the density matrix being initialized does not have self.n_qubits

        """
        if isinstance(data, Graph) or isinstance(data, nx.Graph):
            dm = DensityMatrix.from_graph(data)
        else:
            dm = DensityMatrix(data)

        assert dm.data.shape[0] == dm.data.shape[1] == 2**self.n_qubits
        return dm

    def _initialize_graph(self, data):
        """
        Initializes a graph state based on the data

        :param data: data to construct the Graph representation
        :type data: nx.Graph OR int OR frozenset
        :raises AssertionError: if the graph being initialized does not have self.n_qubits

        """
        graph = Graph(data)

        assert graph.n_qubits == self.n_qubits, (
            f"Expected {self.n_qubits} qubits, "
            f"graph representation has {graph.n_qubits}"
        )
        return graph

    def _initialize_stabilizer(self, data):
        """
        Initializes a stabilizer state based on the data

        :param data: data to construct the stabilizer state representation
        :type data: int or CliffordTableau
        """
        if not self.mixed:
            if isinstance(data, Stabilizer):
                return data
            else:
                stabilizer = Stabilizer(data)
        else:
            if isinstance(data, MixedStabilizer):
                return data
            else:
                stabilizer = MixedStabilizer(data)

        assert stabilizer.n_qubits == self.n_qubits, (
            f"Expected {self.n_qubits} qubits, "
            f"Stabilizer representation has {stabilizer.n_qubits}"
        )
        return stabilizer

    def _initialize_representation(self, representation, data):
        """
        Helper function to initialize any given representation

        :param representation: representation to initialize
        :type representation: str
        :param data: data with which the representation should be initialized
        :type data: int OR frozenset OR Graph OR nx.Graph OR numpy.ndarray
        :raises ValueError: if representation is invalid
        """
        if representation == "dm":
            if self.n_qubits > DENSITY_MATRIX_QUBIT_THRESH:
                warnings.warn(
                    UserWarning(
                        "Density matrix is not recommended for a state of this size"
                    )
                )
            return self._initialize_dm(data)
        elif representation == "graph":
            return self._initialize_graph(data)
        elif representation == "stab":
            return self._initialize_stabilizer(data)
        else:
            raise ValueError("Passed representation is invalid")

    def _density_to_graph(self, rep):
        new_data = rc.density_to_graph(rep.data)
        if isinstance(new_data, list):
            new_rep = MixedGraph(new_data)
        else:
            new_rep = Graph(new_data)
        return new_rep

    def _density_to_stabilizer(self, rep):
        new_data = rc.density_to_stabilizer(rep.data)
        if self.mixed:
            new_tab_list = []
            for p_i, s_i in new_data:
                new_tab_list.append((p_i, CliffordTableau(s_i)))

            new_rep = MixedStabilizer(new_tab_list)
        else:
            new_tableau = CliffordTableau(new_data[0][1])
            new_rep = Stabilizer(new_tableau)
        return new_rep

    def _stabilizer_to_density(self, rep):
        """

        :param rep:
        :type rep: Stabilizer or MixedStabilizer
        :return:
        :rtype:
        """
        data = rep.data
        if self.mixed:
            data_list = []
            for (p_i, t_i) in data.mixture:
                data_list.append((p_i, t_i.to_stabilizer()))
            rho = rc.stabilizer_to_density(data_list)
        else:
            rho = rc.stabilizer_to_density(data.to_stabilizer())
        return DensityMatrix(rho)

    def _stabilizer_to_graph(self, rep):
        data = rep.data
        if self.mixed:
            data_list = []
            for (p_i, t_i) in data.mixture:
                data_list.append((p_i, t_i.data))
            graph_list = rc.stabilizer_to_graph(data_list)
            return MixedGraph(graph_list)
        else:
            graph_list = rc.stabilizer_to_graph(data)
            return Graph(graph_list[0][1])

    def _graph_to_density(self, rep):
        rho = rc.graph_to_density(rep.data)
        return DensityMatrix(rho)

    def _graph_to_stabilizer(self, rep):
        new_data = rc.graph_to_stabilizer(rep.data)
        if self.mixed:
            new_tab_list = []
            for p_i, s_i in new_data:
                new_tab_list.append((p_i, CliffordTableau(s_i)))

            new_rep = MixedStabilizer(new_tab_list)
        else:
            new_tableau = CliffordTableau(new_data[0][1])
            new_rep = Stabilizer(new_tableau)
        return new_rep

    def _identity_fun(self, rep):
        return rep

    def convert_representation(self, new_rep_type):
        conversion_dict = {
            ("dm", "graph"): self._density_to_graph,
            ("dm", "stab"): self._density_to_stabilizer,
            ("stab", "dm"): self._stabilizer_to_density,
            ("stab", "graph"): self._stabilizer_to_graph,
            ("graph", "dm"): self._graph_to_density,
            ("graph", "stab"): self._graph_to_stabilizer,
            ("dm", "dm"): self._identity_fun,
            ("stab", "stab"): self._identity_fun,
            ("graph", "graph"): self._identity_fun,
        }
        if self._rep_type != new_rep_type:
            if new_rep_type == "dm" and self.n_qubits > DENSITY_MATRIX_QUBIT_THRESH:
                warnings.warn(
                    UserWarning(
                        "Density matrix is not recommended for a state of this size"
                    )
                )

            tmp_data = self._rep_data
            conversion_func = conversion_dict[(self._rep_type, new_rep_type)]

            self._rep_data = conversion_func(tmp_data)
            self._rep_type = new_rep_type

    @classmethod
    def validate_data(cls, data):
        """
        Validate data type for input data

        :param data: input data
        :type data: int or np.ndarray or CliffordTableau or nx.Graph or frozenset
        :return: True and the number of qubits if the data type is valid
        :rtype: bool, int
        """
        valid = True
        if isinstance(data, int):
            n_qubits = data
        elif isinstance(data, np.ndarray):
            assert (
                data.shape[0] == data.shape[1]
            ), "Input data is a matrix but it is not a square matrix."
            n_qubits = int(np.log2(data.shape[0]))
        elif isinstance(data, CliffordTableau):
            n_qubits = data.n_qubits
        elif isinstance(data, nx.Graph):
            n_qubits = data.number_of_nodes()
        elif isinstance(data, frozenset):
            n_qubits = len(data)
        elif isinstance(data, list):
            if isinstance(data[0][1], CliffordTableau) or isinstance(data[0][1], nx.Graph):
                n_qubits = data[0][1].n_qubits
            else:
                return False, None
        else:
            valid = False
            n_qubits = None
        return valid, n_qubits

    def apply_clifford(self, gate):
        pass
