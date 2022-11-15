import copy
from abc import ABC
import numpy as np
import src.backends.stabilizer.functions.utils as sfu


class TableauBase(ABC):
    """
    The base class for Stabilizer and Clifford Tableau

    """

    def __init__(self, table, phase, n_qubits, shape):
        """
        Initialize a TableauBase object

        :param table: the main matrix of the tableau
        :type table: np.ndarray
        :param phase: the phase vector
        :type phase: np.ndarray
        :param n_qubits:
        :type n_qubits: int
        :param shape: the shape of the tableau
        :type shape: (int, int)
        """
        self._table = table
        self._phase = phase
        self.n_qubits = n_qubits
        self.shape = shape

    @property
    def table(self):
        """
        The tableau data

        :return: the table
        :rtype: numpy.ndarray
        """
        return self._table

    @table.setter
    def table(self, value):
        """
        Set the tableau data

        :param value: the new table
        :type value: numpy.ndarray
        :return: nothing
        :rtype: None
        """
        assert value.shape == self.shape
        self._table = value.astype(int)

    @property
    def phase(self):
        """
        The phase vector

        :return: the phase vector
        :rtype: numpy.ndarray
        """
        return self._phase

    @phase.setter
    def phase(self, value):
        """
        Set the phase vector

        :param value: the new phase vector
        :type value: numpy.ndarray
        :return: nothing
        :rtype: None
        """
        assert value.shape[0] == self._phase.shape[0]
        self._phase = value.astype(int)


class StabilizerTableau(TableauBase):
    """
    The stabilizer tableau, which is the binary symplectic representation of stabilizer generators
    """

    def __init__(self, data, phase=None):
        """
        Construct a stabilizer tableau

        :param data: the data used to initialize the stabilizer tableau
        :type data: int or numpy.ndarray or [numpy.ndarray, numpy.ndarray]
        :param phase: the phase vector
        :type phase: numpy.ndarray
        """
        if isinstance(data, int):
            self._table = np.hstack([np.zeros((data, data)), np.eye(data)]).astype(int)
            self.n_qubits = data
        elif isinstance(data, np.ndarray):
            assert 2 * data.shape[0] == data.shape[1]
            self._table = np.copy(data).astype(int)
            self.n_qubits = data.shape[0]
        elif isinstance(data, list):
            assert len(data) == 2
            assert isinstance(data[0], np.ndarray) and isinstance(data[1], np.ndarray)
            assert data[0].shape == data[1].shape
            self._table = np.hstack(data).astype(int)
            self.n_qubits = data[0].shape[1]
        else:
            raise TypeError("Cannot support the input type")

        if isinstance(phase, np.ndarray) and phase.shape[0] == self.n_qubits:
            self._phase = np.copy(phase).astype(int)
        else:
            self._phase = np.zeros(self.n_qubits).astype(int)

        self.shape = (self.n_qubits, 2 * self.n_qubits)

    def copy(self):
        return StabilizerTableau(self._table, self._phase)

    @property
    def x_matrix(self):
        """
        The X part of the binary symplectic representation

        :return: the table that contains stabilizer generators for X part
        :rtype: numpy.ndarray
        """
        return self._table[:, 0 : self.n_qubits]

    @x_matrix.setter
    def x_matrix(self, value):
        """
        Set the X part of the binary symplectic representation

        :param value: the X matrix part of the stabilizer tableau
        :type value: numpy.ndarray
        :return:
        """
        assert value.shape == (self.n_qubits, self.n_qubits)
        self._table[:, 0 : self.n_qubits] = value.astype(int)

    @property
    def z_matrix(self):
        """
        The Z part of the binary symplectic representation

        :return: the table that contains stabilizer generators for Z part
        :rtype: numpy.ndarray
        """
        return self._table[:, self.n_qubits : 2 * self.n_qubits]

    @z_matrix.setter
    def z_matrix(self, value):
        """
        Set the Z part of the binary symplectic representation

        :param value: the Z matrix part of the stabilizer tableau
        :type value: numpy.ndarray
        :return: nothing
        :rtype: None
        """
        assert value.shape == (self.n_qubits, self.n_qubits)
        self._table[:, self.n_qubits : 2 * self.n_qubits] = value.astype(int)

    def __str__(self):
        return f"Stabilizer: \n {self.to_labels()} \n Phase: \n {self.phase}"

    def __eq__(self, other):
        if isinstance(other, StabilizerTableau):
            return np.all(self.phase == other.phase) and np.array_equal(
                self.table.astype(int), other.table.astype(int)
            )
        return False

    def to_labels(self):
        """
        Convert the stabilizer tableau to generator strings

        :return: a list of strings that represent the stabilizer generators
        :rtype: list[str]
        """
        return sfu.symplectic_to_string(self.x_matrix, self.z_matrix)

    def from_labels(self, labels):
        """
        Set the Stabilizer tableau according to the generator strings

        :param labels: a list of strings that represent the stabilizer generators
        :type labels: list[str]
        :return: nothing
        :rtype: None
        """
        self.x_matrix, self.z_matrix = sfu.string_to_symplectic(labels)

    def validate(self):
        """
        Validate that the tableau is a valid tableau

        :return: True if the tableau is symplectic; False if it is not
        :rtype: bool
        """
        return sfu.is_stabilizer(self._table)

    def _reset(self, new_table, new_phase):
        new_n_qubits = int(new_table.shape[0])
        self._table = new_table.astype(int)
        self._phase = new_phase.astype(int)
        self.n_qubits = new_n_qubits
        self.shape = (self.n_qubits, 2 * self.n_qubits)

    def expand(self, new_table, new_phase):
        """
        Expand the tableau by adding more qubits

        :param new_table: a new table that represents the stabilizer generators
        :type new_table: numpy.ndarray
        :param new_phase: a new phase vector for :math:`-1` phase exponent
        :type new_phase: numpy.ndarray
        :return: nothing
        :rtype: None
        """
        new_n_qubits = int(new_table.shape[0])
        assert new_n_qubits > self.n_qubits
        self._reset(new_table, new_phase)

    def shrink(self, new_table, new_phase):
        """
        Shrink the tableau by removing qubits

        :param new_table: a new table that represents the stabilizer generators
        :type new_table: numpy.ndarray
        :param new_phase: a new phase vector for :math:`-1` phase exponent
        :type new_phase: numpy.ndarray
        :return: nothing
        :rtype: None
        """
        new_n_qubits = int(new_table.shape[0])
        assert new_n_qubits < self.n_qubits
        self._reset(new_table, new_phase)


class CliffordTableau(TableauBase):
    def __init__(self, data, phase=None, *args, **kwargs):
        """
        TODO: support more ways to initialize the tableau

        :param data:
        :type data: numpy.ndarray or int or CliffordTableau
        :param phase: a phase vector
        :type phase: numpy.ndarray
        """
        if isinstance(data, int):
            self._table = np.eye(2 * data).astype(int)
            self.n_qubits = data
            self._initialize_phase(phase)
        elif isinstance(data, np.ndarray):
            assert data.shape[0] == data.shape[1]
            self._table = data.astype(int)
            self.n_qubits = int(data.shape[1] / 2)
            self._initialize_phase(phase)
        elif isinstance(data, CliffordTableau):
            self._table = np.copy(data.table)
            self.n_qubits = data.n_qubits
            self._phase = np.copy(data.phase)
            self._iphase = np.copy(data.iphase)
        else:
            raise TypeError("Cannot support the input type")
        self.shape = (2 * self.n_qubits, 2 * self.n_qubits)

    def _initialize_phase(self, phase):
        if isinstance(phase, np.ndarray) and phase.shape[0] == 2 * self.n_qubits:
            self._phase = phase.astype(int)
        else:
            self._phase = np.zeros(2 * self.n_qubits).astype(int)
        self._iphase = np.zeros(2 * self.n_qubits).astype(int)

    def copy(self):
        return copy.deepcopy(self)

    @property
    def table_x(self):
        """
        The X part of the tableau

        :return: the table that contains destabilizer and stabilizer generators for X part
        :rtype: numpy.ndarray
        """
        return self._table[:, 0 : self.n_qubits]

    @table_x.setter
    def table_x(self, value):
        """
        Set the X part of the tableau

        :param value: the new matrix for the X part of the tableau
        :type value: numpy.ndarray
        :return: nothing
        :rtype: None
        """
        assert value.shape == (2 * self.n_qubits, self.n_qubits)
        value = value.astype(int)
        self._table[:, 0 : self.n_qubits] = value

    @property
    def table_z(self):
        """
        The Z part of the tableau

        :return: the table that contains destabilizer and stabilizer generators of the Z part
        :rtype: numpy.ndarray
        """
        return self._table[:, self.n_qubits : 2 * self.n_qubits]

    @table_z.setter
    def table_z(self, value):
        """
        Set the Z part of the tableau

        :param value: the new matrix for the Z part of the tableau
        :type value: numpy.ndarray
        :return: nothing
        :rtype: None
        """
        assert value.shape == (2 * self.n_qubits, self.n_qubits)
        value = value.astype(int)
        self._table[:, self.n_qubits : 2 * self.n_qubits] = value

    @property
    def destabilizer(self):
        """
        The destabilizer part of the tableau

        :return: the destabilizer part of the tableau
        :rtype: numpy.ndarray
        """
        return self._table[0 : self.n_qubits]

    @destabilizer.setter
    def destabilizer(self, value):
        """
        Set the destabilizer part of the tableau

        :param value: a new matrix for the destabilizer part of the tableau
        :type value: numpy.ndarray
        :return: nothing
        :rtype: None
        """

        assert value.shape == (self.n_qubits, 2 * self.n_qubits)
        self._table[0 : self.n_qubits] = value.astype(int)

    @property
    def destabilizer_x(self):
        """
        The X part of the destabilizer in the tableau

        :return: the X part of the destabilizer in the tableau
        :rtype: numpy.ndarray
        """
        return self._table[0 : self.n_qubits, 0 : self.n_qubits]

    @destabilizer_x.setter
    def destabilizer_x(self, value):
        """
        Set the X part of the destabilizer in the tableau

        :param value:
        :type value: numpy.ndarray
        :return: nothing
        :rtype: None
        """
        assert value.shape == (self.n_qubits, self.n_qubits)
        self._table[0 : self.n_qubits, 0 : self.n_qubits] = value.astype(int)

    @property
    def destabilizer_z(self):
        """
        The Z part of the destabilizer in the tableau

        :return: the Z part of the destabilizer in the tableau
        :rtype: numpy.ndarray
        """
        return self._table[0 : self.n_qubits, self.n_qubits : 2 * self.n_qubits]

    @destabilizer_z.setter
    def destabilizer_z(self, value):
        """
        Set the Z part of the destabilizer in the tableau

        :param value: the Z part of the destabilizer in the tableau
        :type value: numpy.ndarray
        :return: nothing
        :rtype: None
        """
        assert value.shape == (self.n_qubits, self.n_qubits)
        value = value.astype(int)
        self._table[0 : self.n_qubits, self.n_qubits : 2 * self.n_qubits] = value

    @property
    def stabilizer(self):
        """
        The stabilizer part of the tableau

        :return: the stabilizer part of the tableau
        :rtype: numpy.ndarray
        """
        return self._table[self.n_qubits :]

    @stabilizer.setter
    def stabilizer(self, value):
        """
        Set the stabilizer part of the tableau

        :param value: a new matrix that represents the stabilizer part of the tableau
        :type value: np.ndarray
        :return: nothing
        :rtype: None
        """
        assert sfu.is_symplectic_self_orthogonal(value)
        assert value.shape == (self.n_qubits, 2 * self.n_qubits)
        self._table[self.n_qubits :] = value.astype(int)

    @property
    def stabilizer_x(self):
        """
        The X part of the stabilizer in the tableau

        :return: the X part of the stabilizer in the tableau
        :rtype: numpy.ndarray
        """
        return self._table[self.n_qubits :, 0 : self.n_qubits]

    @stabilizer_x.setter
    def stabilizer_x(self, value):
        """
        Set the X part of the stabilizer in the tableau

        :param value: the X part of the stabilizer in the tableau
        :type value: numpy.ndarray
        :return: nothing
        :rtype: None
        """

        assert value.shape == (self.n_qubits, self.n_qubits)
        self._table[self.n_qubits :, 0 : self.n_qubits] = value.astype(int)

    @property
    def stabilizer_z(self):
        """
        The Z part of the stabilizer in the tableau

        :return: the Z part of the stabilizer in the tableau
        :rtype: numpy.ndarray
        """
        return self._table[self.n_qubits :, self.n_qubits : 2 * self.n_qubits]

    @stabilizer_z.setter
    def stabilizer_z(self, value):
        """
        The Z part of the stabilizer in the tableau

        :param value: a new matrix that represents the Z part of the stabilizer in the tableau
        :type value: numpy.ndarray
        :return: nothing
        :rtype: None
        """

        assert value.shape == (self.n_qubits, self.n_qubits)
        value = value.astype(int)
        self._table[self.n_qubits :, self.n_qubits : 2 * self.n_qubits] = value

    @property
    def iphase(self):
        """
        A phase vector that represents the phase of i factor (imaginary part)

        :return: a phase vector that represents the phase of i factor (imaginary part)
        :rtype: numpy.ndarray
        """
        return self._iphase

    @iphase.setter
    def iphase(self, value):
        """
        Set the phase vector that represents the phase of i factor (imaginary part)

        :param value: a new phase vector
        :type value: numpy.ndarray
        :return: nothing
        :rtype: None
        """
        assert value.shape[0] == 2 * self.n_qubits
        self._iphase = value.astype(int)

    def __str__(self):
        return (
            f"Destabilizers: \n{self.destabilizer_to_labels()}\n Stabilizer: \n {self.stabilizer_to_labels()} \n "
            f"Phase: \n {self.phase}\n i phase: \n {self.iphase}"
        )

    def stabilizer_to_labels(self):
        """
        Convert the stabilizers to generator strings

        :return:
        :rtype:
        """
        return sfu.symplectic_to_string(self.stabilizer_x, self.stabilizer_z)

    def destabilizer_to_labels(self):
        """
        Convert the destabilizers to the generator strings

        :return:
        :rtype:
        """
        return sfu.symplectic_to_string(self.destabilizer_x, self.destabilizer_z)

    def stabilizer_from_labels(self, labels):
        """
        Set the stabilizer part of the tableau according to generator strings

        :param labels: generator strings for stabilizers
        :type labels: list[str]
        :return: nothing
        :rtype: None
        """
        self.stabilizer_x, self.stabilizer_z = sfu.string_to_symplectic(labels)

    def destabilizer_from_labels(self, labels):
        """
        Set the destabilizer part of the tableau according to generator strings

        :param labels: generator strings for destabilizers
        :type labels: list[str]
        :return: nothing
        :rtype: None
        """
        self.destabilizer_x, self.destabilizer_z = sfu.string_to_symplectic(labels)

    def __eq__(self, other):
        # check without converting to canonical form
        if isinstance(other, CliffordTableau):
            return (
                np.all(self.phase == other.phase)
                and np.all(self.iphase == other.iphase)
                and np.array_equal(self.table.astype(int), other.table.astype(int))
            )

        return False

    def to_stabilizer(self):
        """
        Return a StabilizerTableau that contains only the stabilizer part of the tableau

        :return: a StabilizerTableau that contains only the stabilizer part
        :rtype: StabilizerTableau
        """
        return StabilizerTableau(self.stabilizer, self.phase[self.n_qubits :])

    def _reset(self, new_table, new_phase, new_iphase):
        new_n_qubits = int(new_table.shape[0] / 2)
        assert len(new_phase) == 2 * new_n_qubits
        self._table = new_table.astype(int)
        self._phase = new_phase.astype(int)
        self._iphase = new_iphase.astype(int)
        self.n_qubits = new_n_qubits
        self.shape = (2 * self.n_qubits, 2 * self.n_qubits)

    def expand(self, new_table, new_phase, new_iphase):
        """
        Expand the tableau by adding more qubits

        :param new_table: a new table that represents the stabilizer and destabilizer generators
        :type new_table: numpy.ndarray
        :param new_phase: a new phase vector for :math:`-1` phase exponent
        :type new_phase: numpy.ndarray
        :param new_iphase: a new phase vector for :math:`i` phase exponent
        :type new_iphase: numpy.ndarray
        :return: nothing
        :rtype: None
        """
        new_n_qubits = int(new_table.shape[0] / 2)
        assert new_n_qubits > self.n_qubits
        self._reset(new_table, new_phase, new_iphase)

    def shrink(self, new_table, new_phase, new_iphase):
        """
        Shrink the tableau by removing qubits

        :param new_table: a new table that represents the stabilizer and destabilizer generators
        :type new_table: numpy.ndarray
        :param new_phase: a new phase vector for :math:`-1` phase exponent
        :type new_phase: numpy.ndarray
        :param new_iphase: a new phase vector for :math:`i` phase exponent
        :type new_iphase: numpy.ndarray
        :return: nothing
        :rtype: None
        """
        new_n_qubits = int(new_table.shape[0] / 2)
        assert new_n_qubits < self.n_qubits
        self._reset(new_table, new_phase, new_iphase)
