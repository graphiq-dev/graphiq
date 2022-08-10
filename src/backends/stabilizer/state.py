"""
Stabilizer state representation
"""
from src.backends.state_base import StateRepresentationBase
import src.backends.stabilizer.functions.clifford as sfc
from src.backends.stabilizer.tableau import CliffordTableau


class Stabilizer(StateRepresentationBase):
    def __init__(self, data, *args, **kwargs):
        super().__init__(data, *args, **kwargs)
        if isinstance(data, int):
            self._tableau = CliffordTableau(data)
        elif isinstance(data, CliffordTableau):
            self._tableau = data
        else:
            raise TypeError(
                f"Cannot initialize the stabilizer representation with datatype: {type(data)}"
            )

    @classmethod
    def valid_datatype(cls, data):
        return isinstance(data, (int, CliffordTableau))

    @property
    def n_qubit(self):
        """
        Returns the number of qubits in the stabilizer state

        :return: the number of qubits in the state
        :rtype: int
        """
        return self.tableau.n_qubits

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
        raise NotImplementedError(
            "Stabilizer backend does not support general unitary operation."
        )

    def apply_measurement(
        self, qubit_position, measurement_determinism="probabilistic"
    ):
        """

        :param qubit_position:
        :type qubit_position:
        :param measurement_determinism: if "probabilistic", measurement results are probabilistically selected
                                    if 1, measurement results default to 1 unless the probability of measuring p(1) = 0
                                    if 0, measurement results default to 0 unless the probability of measuring p(0) = 0
        :type measurement_determinism: str/int
        :return:
        :rtype:
        """
        self._tableau, outcome, _ = sfc.z_measurement_gate(
            self._tableau, qubit_position, measurement_determinism
        )
        return outcome

    def apply_hadamard(self, qubit_position):
        self._tableau = sfc.hadamard_gate(self._tableau, qubit_position)

    def apply_cnot(self, control, target):
        self._tableau = sfc.cnot_gate(self._tableau, control, target)

    def apply_cphase(self, control, target):
        self._tableau = sfc.control_z_gate(self._tableau, control, target)

    def apply_phase(self, qubit_position):
        self._tableau = sfc.phase_gate(self._tableau, qubit_position)

    def apply_sigmax(self, qubit_position):
        self._tableau = sfc.x_gate(self._tableau, qubit_position)

    def apply_sigmay(self, qubit_position):
        self._tableau = sfc.y_gate(self._tableau, qubit_position)

    def apply_sigmaz(self, qubit_position):
        self._tableau = sfc.z_gate(self._tableau, qubit_position)

    def reset_qubit(self, qubit_position, measurement_determinism="probabilistic"):
        self._tableau = sfc.reset_z(
            self._tableau, qubit_position, 0, measurement_determinism
        )

    def remove_qubit(self, qubit_position):
        self._tableau = sfc.remove_qubit(self._tableau, qubit_position)
