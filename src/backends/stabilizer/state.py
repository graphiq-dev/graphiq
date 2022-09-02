"""
State representation using the stabilizer formalism
"""
import src.backends.stabilizer.functions.transformation as transform
from src.backends.state_base import StateRepresentationBase
import src.backends.stabilizer.functions.clifford as sfc
from src.backends.stabilizer.tableau import CliffordTableau
from src.backends.stabilizer.functions.stabilizer import canonical_form


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
        return self._tableau.n_qubits

    @property
    def tableau(self):
        """

        :return: the underlying representation
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

    @property
    def data(self):
        """

        :return:
        :rtype:
        """
        return self.tableau

    @data.setter
    def data(self, value):
        """

        :param value:
        :type value:
        :return:
        :rtype:
        """
        self.tableau = value

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
        self._tableau, outcome, _, = sfc.z_measurement_gate(
            self._tableau, qubit_position, measurement_determinism
        )
        return outcome

    def apply_hadamard(self, qubit_position):
        self._tableau = transform.hadamard_gate(self._tableau, qubit_position)

    def apply_cnot(self, control, target):
        self._tableau = transform.cnot_gate(self._tableau, control, target)

    def apply_cz(self, control, target):
        self._tableau = transform.control_z_gate(self._tableau, control, target)

    def apply_phase(self, qubit_position):
        self._tableau = transform.phase_gate(self._tableau, qubit_position)

    def apply_sigmax(self, qubit_position):
        self._tableau = transform.x_gate(self._tableau, qubit_position)

    def apply_sigmay(self, qubit_position):
        self._tableau = transform.y_gate(self._tableau, qubit_position)

    def apply_sigmaz(self, qubit_position):
        self._tableau = transform.z_gate(self._tableau, qubit_position)

    def reset_qubit(self, qubit_position, measurement_determinism="probabilistic"):
        self._tableau = sfc.reset_z(
            self._tableau, qubit_position, 0, measurement_determinism
        )

    def remove_qubit(self, qubit_position, measurement_determinism="probabilistic"):
        self._tableau = sfc.remove_qubit(
            self._tableau, qubit_position, measurement_determinism
        )

    def trace_out_qubits(
        self, qubit_positions, measurement_determinism="probabilistic"
    ):
        self._tableau = sfc.partial_trace(
            self._tableau,
            keep=qubit_positions,
            dims=self.n_qubit * [2],
            measurement_determinism=measurement_determinism,
        )

    def __str__(self):
        """

        :return:
        :rtype: str
        """
        return self._tableau.__str__()

    def __eq__(self, other):
        """

        :param other:
        :type other: Stabilizer
        :return: True if the stabilizer tableaux of two Stabilizer objects are the same
        :rtype: bool
        """
        tableau1 = canonical_form(self.data.to_stabilizer())
        tableau2 = canonical_form(other.data.to_stabilizer())
        return tableau1 == tableau2
