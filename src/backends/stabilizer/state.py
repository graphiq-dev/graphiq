import numpy as np

from src.backends.state_base import StateRepresentationBase
import src.backends.stabilizer.functions as sf
from abc import ABC


class CliffordTableau(ABC):
    def __init__(self, data, *args, **kwargs):
        """
        TODO: support more ways to initialize the tableau

        :param data:
        :type data:
        """
        if isinstance(data, int):

            self._table = np.block(
                [
                    [np.eye(data), np.zeros((data, data))],
                    [np.zeros((data, data)), np.eye(data)],
                ]
            ).astype(int)
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
        assert sf.is_symplectic_self_orthogonal(value)
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
        return sf.symplectic_to_string(self.stabilizer_x, self.stabilizer_z)

    def destabilizer_to_labels(self):
        """

        :return:
        :rtype:
        """
        return sf.symplectic_to_string(self.destabilizer_x, self.destabilizer_z)

    def stabilizer_from_labels(self, labels):
        """

        :param labels:
        :type labels: list[str]
        :return:
        :rtype: None
        """
        self.stabilizer_x, self.stabilizer_z = sf.string_to_symplectic(labels)

    def destabilizer_from_labels(self, labels):
        """

        :param labels:
        :type labels: list[str]
        :return:
        :rtype: None
        """
        self.destabilizer_x, self.destabilizer_z = sf.string_to_symplectic(labels)


class Stabilizer(StateRepresentationBase):
    def __init__(self, data, *args, **kwargs):
        super().__init__(data, *args, **kwargs)
        if isinstance(data, int):
            self._tableau = CliffordTableau(data)
        elif isinstance(data, CliffordTableau):
            self._tableau = data
        else:
            raise TypeError("Cannot initialize the stabilizer representation")

    @property
    def tableau(self):
        """

        :return:
        :rtype: CliffordTableau
        """
        return self._tableau

    @tableau.setter
    def tableau(self, value):
        """

        :param value:
        :type value: int or CliffordTableau
        :return: nothing
        :rtype: None
        """
        if isinstance(value, int):
            self._tableau = CliffordTableau(value)
        elif isinstance(value, CliffordTableau):
            self._tableau = value
        else:
            raise TypeError("Must use CliffordTableau for the stabilizer's tableau")

    def apply_unitary(self, qubit_position, unitary):
        pass

    def apply_measurements(self, qubit_position):
        pass

    def apply_Hadamard(self, qubit_position):
        pass

    def apply_CNOT(self, control, target):
        pass

    def apply_CPhase(self, control, target):
        pass

    def apply_Phase(self, qubit_position):
        pass

    def apply_SigmaX(self, qubit_position):
        pass

    def apply_SigmaY(self, qubit_position):
        pass

    def apply_SigmaZ(self, qubit_position):
        pass
