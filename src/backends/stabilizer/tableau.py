from abc import ABC
import numpy as np
import src.backends.stabilizer.functions.utils as sfu


class TableauBase(ABC):
    """
    The base class for Stabilizer and Clifford Tableau

    """

    def __init__(self, table, phase, n_qubits, shape):
        self._table = table
        self._phase = phase
        self.n_qubits = n_qubits
        self.shape = shape

    @property
    def table(self):
        """

        :return: the table
        :rtype: numpy.ndarray
        """
        return self._table

    @table.setter
    def table(self, value):
        """

        :param value:
        :return:
        """
        assert value.shape == self.shape
        self._table = value.astype(int)

    @property
    def phase(self):
        """

        :return:
        :rtype:
        """
        return self._phase

    @phase.setter
    def phase(self, value):
        """

        :param value:
        :type value:
        :return:
        :rtype:
        """
        assert value.shape[0] == self.shape[0]
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

    @property
    def x_matrix(self):
        """

        :return: the table that contains stabilizer generators for X part
        :rtype: numpy.ndarray
        """
        return self._table[:, 0 : self.n_qubits]

    @x_matrix.setter
    def x_matrix(self, value):
        """

        :param value: the X matrix part of the stabilizer tableau
        :type value: numpy.ndarray
        :return:
        """
        assert value.shape == (self.n_qubits, self.n_qubits)
        self._table[:, 0 : self.n_qubits] = value.astype(int)

    @property
    def z_matrix(self):
        """

        :return: the table that contains stabilizer generators for Z part
        :rtype: numpy.ndarray
        """
        return self._table[:, self.n_qubits : 2 * self.n_qubits]

    @z_matrix.setter
    def z_matrix(self, value):
        """

        :param value:
        :return:
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

        :return:
        :rtype:
        """
        return sfu.symplectic_to_string(self.x_matrix, self.z_matrix)

    def from_labels(self, labels):
        """

        :param labels:
        :type labels: list[str]
        :return:
        :rtype: None
        """
        self.x_matrix, self.z_matrix = sfu.string_to_symplectic(labels)

    def validate(self):
        return sfu.is_stabilizer(self._table)


class CliffordTableau(TableauBase):
    def __init__(self, data, phase=None, *args, **kwargs):
        """
        TODO: support more ways to initialize the tableau

        :param data:
        :type data:
        :param phase:
        :type phase:
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

    @property
    def table_x(self):
        """
        :return: the table that contains destabilizer and stabilizer generators for X part
        :rtype: numpy.ndarray
        """
        return self._table[:, 0 : self.n_qubits]

    @table_x.setter
    def table_x(self, value):
        """
        :param value:
        :return:
        """
        assert value.shape == (2 * self.n_qubits, self.n_qubits)
        value = value.astype(int)
        self._table[:, 0 : self.n_qubits] = value

    @property
    def table_z(self):
        """
        :return: the table that contains destabilizer and stabilizer generators
        :rtype: numpy.ndarray
        """
        return self._table[:, self.n_qubits : 2 * self.n_qubits]

    @table_z.setter
    def table_z(self, value):
        """
        :param value:
        :return:
        """
        assert value.shape == (2 * self.n_qubits, self.n_qubits)
        value = value.astype(int)
        self._table[:, self.n_qubits : 2 * self.n_qubits] = value

    @property
    def destabilizer(self):
        """
        :return:
        :rtype:
        """
        return self._table[0 : self.n_qubits]

    @destabilizer.setter
    def destabilizer(self, value):
        """
        :param value:
        :type value:
        :return:
        :rtype:
        """

        assert value.shape == (self.n_qubits, 2 * self.n_qubits)
        self._table[0 : self.n_qubits] = value.astype(int)

    @property
    def destabilizer_x(self):
        """
        :return:
        :rtype:
        """
        return self._table[0 : self.n_qubits, 0 : self.n_qubits]

    @destabilizer_x.setter
    def destabilizer_x(self, value):
        """
        :param value:
        :type value:
        :return:
        :rtype:
        """
        assert value.shape == (self.n_qubits, self.n_qubits)
        self._table[0 : self.n_qubits, 0 : self.n_qubits] = value.astype(int)

    @property
    def destabilizer_z(self):
        """
        :return:
        :rtype:
        """
        return self._table[0 : self.n_qubits, self.n_qubits : 2 * self.n_qubits]

    @destabilizer_z.setter
    def destabilizer_z(self, value):
        """
        :param value:
        :type value:
        :return:
        :rtype:
        """
        assert value.shape == (self.n_qubits, self.n_qubits)
        value = value.astype(int)
        self._table[0 : self.n_qubits, self.n_qubits : 2 * self.n_qubits] = value

    @property
    def stabilizer(self):
        """
        :return:
        :rtype:
        """
        return self._table[self.n_qubits :]

    @stabilizer.setter
    def stabilizer(self, value):
        """
        :param value:
        :type value: np.ndarray
        :return:
        :rtype:
        """
        assert sfu.is_symplectic_self_orthogonal(value)
        assert value.shape == (self.n_qubits, 2 * self.n_qubits)
        self._table[self.n_qubits :] = value.astype(int)

    @property
    def stabilizer_x(self):
        """
        :return:
        :rtype:
        """
        return self._table[self.n_qubits :, 0 : self.n_qubits]

    @stabilizer_x.setter
    def stabilizer_x(self, value):
        """
        :param value:
        :type value:
        :return:
        :rtype:
        """

        assert value.shape == (self.n_qubits, self.n_qubits)
        self._table[self.n_qubits :, 0 : self.n_qubits] = value.astype(int)

    @property
    def stabilizer_z(self):
        """
        :return:
        :rtype:
        """
        return self._table[self.n_qubits :, self.n_qubits : 2 * self.n_qubits]

    @stabilizer_z.setter
    def stabilizer_z(self, value):
        """
        :param value:
        :type value:
        :return:
        :rtype:
        """

        assert value.shape == (self.n_qubits, self.n_qubits)
        value = value.astype(int)
        self._table[self.n_qubits :, self.n_qubits : 2 * self.n_qubits] = value

    @property
    def iphase(self):
        """
        :return:
        :rtype:
        """
        return self._iphase

    @iphase.setter
    def iphase(self, value):
        """
        :param value:
        :type value:
        :return:
        :rtype:
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
        :return:
        :rtype:
        """
        return sfu.symplectic_to_string(self.stabilizer_x, self.stabilizer_z)

    def destabilizer_to_labels(self):
        """
        :return:
        :rtype:
        """
        return sfu.symplectic_to_string(self.destabilizer_x, self.destabilizer_z)

    def stabilizer_from_labels(self, labels):
        """
        :param labels:
        :type labels: list[str]
        :return:
        :rtype: None
        """
        self.stabilizer_x, self.stabilizer_z = sfu.string_to_symplectic(labels)

    def destabilizer_from_labels(self, labels):
        """
        :param labels:
        :type labels: list[str]
        :return:
        :rtype: None
        """
        self.destabilizer_x, self.destabilizer_z = sfu.string_to_symplectic(labels)

    def __eq__(self, other):
        # TODO: check if it is necessary to reduce to the echelon gauge before comparison
        if isinstance(other, CliffordTableau):
            return (
                np.all(self.phase == other.phase)
                and np.all(self.iphase == other.iphase)
                and np.array_equal(self.table.astype(int), other.table.astype(int))
            )

        return False

    def to_stabilizer(self):
        """

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

    def expand(self, new_table, new_phase, new_iphase):
        new_n_qubits = int(new_table.shape[0] / 2)
        assert new_n_qubits > self.n_qubits
        self._reset(new_table, new_phase, new_iphase)

    def shrink(self, new_table, new_phase, new_iphase):
        new_n_qubits = int(new_table.shape[0] / 2)
        assert new_n_qubits < self.n_qubits
        self._reset(new_table, new_phase, new_iphase)
