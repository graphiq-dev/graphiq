import copy

import numpy as np

import graphiq.backends.stabilizer as stab
from graphiq.backends.stabilizer.functions import utils as sfu
from graphiq.backends.stabilizer.tableau import TableauBase, StabilizerTableau


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
        elif isinstance(data, CliffordTableau) or isinstance(data, StabilizerTableau):
            if isinstance(data, StabilizerTableau):
                data = stab.clifford_from_stabilizer(data)
            self._table = np.copy(data.table)
            self.n_qubits = data.n_qubits
            self._phase = np.copy(data.phase)
            self._iphase = np.copy(data.iphase)

        else:
            raise TypeError(f"Cannot support the input type of {type(data)}")
        self.shape = (2 * self.n_qubits, 2 * self.n_qubits)

    def _initialize_phase(self, phase):
        """
        Helper function to initialize the phase vectors

        :param phase:
        :type phase: numpy.ndarray
        :return: nothing
        :rtype: None
        """
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
