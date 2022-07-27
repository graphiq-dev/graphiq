import numpy as np
from abc import ABC
import src.backends.stabilizer.functions.conversion as sfc


class CliffordTableau(ABC):
    def __init__(self, data, *args, **kwargs):
        """
        TODO: support more ways to initialize the tableau

        :param data:
        :type data:
        """
        if isinstance(data, int):

            self._table = np.eye(2 * data).astype(int)
            self._phase = np.zeros((2 * data, 1)).astype(int)
        elif isinstance(data, np.ndarray):
            assert data.shape[0] == data.shape[1]
            self._table = data.astype(int)
            self._phase = np.zeros((data.shape[0], 1)).astype(int)
        else:
            raise TypeError("Cannot support the input type")

        self.n_qubits = int(self._table.shape[0] / 2)

    @property
    def table(self):
        """

        :return: the table that contains destabilizer and stabilizer generators
        :rtype: numpy.ndarray
        """
        return self._table

    @table.setter
    def table(self, value):
        """

        :param value:
        :return:
        """
        assert value.shape == (2 * self.n_qubits, 2 * self.n_qubits)
        self._table = value

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
        self._table[0 : self.n_qubits] = value

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
        self._table[0 : self.n_qubits, 0 : self.n_qubits] = value

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
        :type value:
        :return:
        :rtype:
        """
        assert sfc.is_symplectic_self_orthogonal(value)
        assert value.shape == (self.n_qubits, 2 * self.n_qubits)
        self._table[self.n_qubits :] = value

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
        self._table[self.n_qubits :, 0 : self.n_qubits] = value

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
        self._table[self.n_qubits :, self.n_qubits : 2 * self.n_qubits] = value

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
        assert value.shape == (2 * self.n_qubits, 1)
        self._phase = value

    def __str__(self):
        return f"Destabilizers: \n{self.destabilizer}\n Stabilizer: \n {self.stabilizer} \n Phase: \n {self.phase}"

    def stabilizer_to_labels(self):
        """

        :return:
        :rtype:
        """
        return sfc.symplectic_to_string(self.stabilizer_x, self.stabilizer_z)

    def destabilizer_to_labels(self):
        """

        :return:
        :rtype:
        """
        return sfc.symplectic_to_string(self.destabilizer_x, self.destabilizer_z)

    def stabilizer_from_labels(self, labels):
        """

        :param labels:
        :type labels: list[str]
        :return:
        :rtype: None
        """
        self.stabilizer_x, self.stabilizer_z = sfc.string_to_symplectic(labels)

    def destabilizer_from_labels(self, labels):
        """

        :param labels:
        :type labels: list[str]
        :return:
        :rtype: None
        """
        self.destabilizer_x, self.destabilizer_z = sfc.string_to_symplectic(labels)
