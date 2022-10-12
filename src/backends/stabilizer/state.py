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
        The data that represents the state given by this Stabilizer representation

        :return: the underlying representation
        :rtype: CliffordTableau
        """
        return self._tableau

    @tableau.setter
    def tableau(self, value):
        """
        Set the data that represents the state given by this Stabilizer representation

        :param value: a new tableau or a parameter to initialize a new tableau
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
        The data that represents the state given by this Stabilizer representation

        :return: the tableau that represents this state
        :rtype: CliffordTableau
        """
        return self.tableau

    @data.setter
    def data(self, value):
        """
        Set the data that represents the state given by this Stabilizer representation

        :param value: a new tableau or a parameter to initialize a new tableau
        :type value: CliffordTableau or int
        :return: nothing
        :rtype: None
        """
        self.tableau = value

    def apply_unitary(self, qubit_position, unitary):
        raise NotImplementedError(
            "Stabilizer backend does not support general unitary operation."
        )

    def apply_circuit(self, gate_list_str, reverse=False):
        """
        Apply a quantum circuit to the tableau

        :param gate_list_str: a list of gates in the circuit
        :type gate_list_str: list[tuple]
        :param reverse: a parameter to indicate whether running the inverse circuit
        :type reverse: bool
        :return: nothing
        :rtype: None
        """
        transform.run_circuit(self._tableau, gate_list_str, reverse=reverse)

    def apply_measurement(
        self, qubit_position, measurement_determinism="probabilistic"
    ):
        """
        Apply the measurement in the computational basis to a given qubit

        :param qubit_position: the qubit position where the measurement is applied
        :type qubit_position: int
        :param measurement_determinism: if "probabilistic", measurement results are probabilistically selected
                if 1, measurement results default to 1 unless the probability of measuring p(1) = 0
                if 0, measurement results default to 0 unless the probability of measuring p(0) = 0
        :type measurement_determinism: str/int
        :return: the measurement outcome
        :rtype: int
        """
        self._tableau, outcome, _, = sfc.z_measurement_gate(
            self._tableau, qubit_position, measurement_determinism
        )
        return outcome

    def apply_hadamard(self, qubit_position):
        """
        Apply the Hadamard gate to the Stabilizer

        :param qubit_position: the qubit position where the gate is applied
        :type qubit_position: int
        :return: nothing
        :rtype: None
        """
        self._tableau = transform.hadamard_gate(self._tableau, qubit_position)

    def apply_cnot(self, control, target):
        """
        Apply CNOT gate to the Stabilizer

        :param control: the control qubit position where the gate is applied
        :type control: int
        :param target: the target qubit position where the gate is applied
        :type target: int
        :return: nothing
        :rtype: None
        """
        self._tableau = transform.cnot_gate(self._tableau, control, target)

    def apply_cz(self, control, target):
        """
        Apply CZ gate to the Stabilizer

        :param control: the control qubit position where the gate is applied
        :type control: int
        :param target: the target qubit position where the gate is applied
        :type target: int
        :return: nothing
        :rtype: None
        """
        self._tableau = transform.control_z_gate(self._tableau, control, target)

    def apply_phase(self, qubit_position):
        """
        Apply the phase gate to the Stabilizer

        :param qubit_position: the qubit position where the gate is applied
        :type qubit_position: int
        :return: nothing
        :rtype: None
        """
        self._tableau = transform.phase_gate(self._tableau, qubit_position)

    def apply_sigmax(self, qubit_position):
        """
        Apply the X gate to the Stabilizer

        :param qubit_position: the qubit position where the gate is applied
        :type qubit_position: int
        :return: nothing
        :rtype: None
        """
        self._tableau = transform.x_gate(self._tableau, qubit_position)

    def apply_sigmay(self, qubit_position):
        """
        Apply the Y gate to the Stabilizer

        :param qubit_position: the qubit position where the gate is applied
        :type qubit_position: int
        :return: nothing
        :rtype: None
        """
        self._tableau = transform.y_gate(self._tableau, qubit_position)

    def apply_sigmaz(self, qubit_position):
        """
        Apply the Z gate to the Stabilizer

        :param qubit_position: the qubit position where the gate is applied
        :type qubit_position: int
        :return: nothing
        :rtype: None
        """
        self._tableau = transform.z_gate(self._tableau, qubit_position)

    def reset_qubit(self, qubit_position, measurement_determinism="probabilistic"):
        """
        Reset a given qubit to :math:`|0\\rangle` state after disentangling it from the rest

        :param qubit_position: the qubit position to be reset
        :type qubit_position: int
        :param measurement_determinism: if "probabilistic", measurement results are probabilistically selected
                if 1, measurement results default to 1 unless the probability of measuring p(1) = 0
                if 0, measurement results default to 0 unless the probability of measuring p(0) = 0
        :type measurement_determinism: str/int
        :return: nothing
        :rtype: None
        """
        self._tableau = sfc.reset_z(
            self._tableau, qubit_position, 0, measurement_determinism
        )

    def remove_qubit(self, qubit_position, measurement_determinism="probabilistic"):
        """
        Trace out one qubit after disentangling it from the rest

        :param qubit_position: the qubit position to be traced out
        :type qubit_position: int
        :param measurement_determinism: if "probabilistic", measurement results are probabilistically selected
                if 1, measurement results default to 1 unless the probability of measuring p(1) = 0
                if 0, measurement results default to 0 unless the probability of measuring p(0) = 0
        :type measurement_determinism: str/int
        :return: nothing
        :rtype: None
        """
        self._tableau = sfc.remove_qubit(
            self._tableau, qubit_position, measurement_determinism
        )

    def trace_out_qubits(
        self, qubit_positions, measurement_determinism="probabilistic"
    ):
        """
        Trace out qubits after disentangling them from the rest

        :param qubit_positions: the qubit positions to be traced out
        :type qubit_positions: list[int]
        :param measurement_determinism: if "probabilistic", measurement results are probabilistically selected
                if 1, measurement results default to 1 unless the probability of measuring p(1) = 0
                if 0, measurement results default to 0 unless the probability of measuring p(0) = 0
        :type measurement_determinism: str/int
        :return: nothing
        :rtype: None
        """
        self._tableau = sfc.partial_trace(
            self._tableau,
            keep=qubit_positions,
            dims=self.n_qubit * [2],
            measurement_determinism=measurement_determinism,
        )

    def __str__(self):
        """
        Return a string representation of this state representation

        :return: a string representation of this state representation
        :rtype: str
        """
        return self._tableau.__str__()

    def __eq__(self, other):
        """
        Compare two Stabilizer objects

        :param other: the other Stabilizer to be compared
        :type other: Stabilizer
        :return: True if the stabilizer tableaux of two Stabilizer objects are the same
        :rtype: bool
        """
        tableau1 = canonical_form(self.data.to_stabilizer())
        tableau2 = canonical_form(other.data.to_stabilizer())
        return tableau1 == tableau2
