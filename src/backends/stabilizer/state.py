"""
State representation using the stabilizer formalism
"""
import copy
import numpy as np

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


class MixedStabilizer(Stabilizer):
    """
    A mixed state representation using the stabilizer formalism, where the mixture is represented as a list of
    pure states (tableaus) and an associated mixture probability.
    """

    def __init__(self, data, *args, **kwargs):
        super().__init__(data, *args, **kwargs)
        if isinstance(data, int):
            self._mixture = [
                (1.0, CliffordTableau(data)),
            ]
        elif isinstance(data, CliffordTableau):
            self._mixture = [
                (1.0, data),
            ]
        elif isinstance(data, list):
            assert all(
                isinstance(pi, float) and isinstance(ti, CliffordTableau)
                for (pi, ti) in data
            )
            self._mixture = data
        else:
            raise TypeError(
                f"Cannot initialize the stabilizer representation with datatype: {type(data)}"
            )

    @classmethod
    def valid_datatype(cls, data):
        valid = isinstance(data, (int, CliffordTableau, list))
        if isinstance(data, list):
            valid = valid and all(
                isinstance(pi, float) and isinstance(ti, CliffordTableau)
                for (pi, ti) in data
            )
        return valid

    @property
    def n_qubit(self):
        """
        Returns the number of qubits in the stabilizer state

        :return: the number of qubits in the state
        :rtype: int
        """
        return self._mixture[0][1].n_qubits

    @property
    def mixture(self):
        """
        The mixture of pure states, represented as a list of tableaus and associated probabilities.

        :return: the mixture as a list of (probability_i, tableau_i)
        :rtype: list
        r"""
        return self._mixture

    @mixture.setter
    def mixture(self, value):
        """
        Sets the mixture of pure states, represented as a list of tableaus and associated probabilities.

        :param value: a new mixture list, pure tableau, or a parameter to initialize a new tableau
        :type value: list or int or CliffordTableau
        :return: the mixture as a list of (probability_i, tableau_i)
        :rtype: list
        r"""
        if isinstance(value, list):
            assert all(
                isinstance(pi, float) and isinstance(ti, CliffordTableau)
                for (pi, ti) in value
            )
            self._mixture = value
        elif isinstance(value, CliffordTableau):
            self._mixture = [(1.0, value)]
        elif isinstance(value, int):
            self._mixture = [(1.0, CliffordTableau(value))]
        else:
            raise TypeError(
                "Must use a list of CliffordTableau for the mixed stabilizer"
            )

    @property
    def data(self):
        """
        The data that represents the state given by this MixedStabilizer representation

        :return: the mixture that represents this state
        :rtype: list
        """
        return self.mixture

    @data.setter
    def data(self, value):
        """
        Set the data that represents the state given by this Stabilizer representation

        :param value: a new tableau or a parameter to initialize a new tableau
        :type value: CliffordTableau or int
        :return: nothing
        :rtype: None
        """
        self.mixture = value

    @property
    def probability(self):
        """
        Computes the total probability as the summed probability of all pure states in the mixture
        :math:`\sum_i p_i \ \forall (p_i, \mathcal{T}_i`.
        :return:
        """
        return sum(pi for pi, ti in self.mixture)

    @property
    def tableau(self):
        return TypeError(
            "Simulating using a mixed state representation, no tableau defined."
        )

    def reduce(self):
        mixture_temp = self.mixture
        # print("starting len", len(mixture_temp))
        mixture_reduce = []
        while len(mixture_temp) != 0:
            p0, t0 = mixture_temp[0]
            mixture_temp.pop(0)
            for i, (pi, ti) in enumerate(mixture_temp):
                if np.count_nonzero(t0 != ti) == 0:
                    # print("same")
                    p0 += pi
                    mixture_temp.pop(i)

            mixture_reduce.append((p0, t0))

        # print(mixture_reduce)
        # print("len after reduction", len(mixture_reduce))
        self.mixture = mixture_reduce

    def apply_unitary(self, qubit_position, unitary):
        raise NotImplementedError(
            "Stabilizer backend does not support general unitary operation."
        )

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
        # TODO: how to best measure?
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
        self._mixture = [
            (pi, transform.hadamard_gate(tableau_i, qubit_position))
            for (pi, tableau_i) in self._mixture
        ]

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
        self._mixture = [
            (pi, transform.cnot_gate(tableau_i, control, target))
            for (pi, tableau_i) in self._mixture
        ]

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
        self._mixture = [
            (pi, transform.control_z_gate(tableau_i, control, target))
            for (pi, tableau_i) in self._mixture
        ]

    def apply_phase(self, qubit_position):
        """
        Apply the phase gate to the Stabilizer

        :param qubit_position: the qubit position where the gate is applied
        :type qubit_position: int
        :return: nothing
        :rtype: None
        """
        self._mixture = [
            (pi, transform.phase_gate(tableau_i, qubit_position))
            for (pi, tableau_i) in self._mixture
        ]

    def apply_sigmax(self, qubit_position):
        """
        Apply the X gate to the Stabilizer

        :param qubit_position: the qubit position where the gate is applied
        :type qubit_position: int
        :return: nothing
        :rtype: None
        """
        self._mixture = [
            (pi, transform.x_gate(tableau_i, qubit_position))
            for (pi, tableau_i) in self._mixture
        ]

    def apply_sigmay(self, qubit_position):
        """
        Apply the Y gate to the Stabilizer

        :param qubit_position: the qubit position where the gate is applied
        :type qubit_position: int
        :return: nothing
        :rtype: None
        """
        self._mixture = [
            (pi, transform.y_gate(tableau_i, qubit_position))
            for (pi, tableau_i) in self._mixture
        ]

    def apply_sigmaz(self, qubit_position):
        """
        Apply the Z gate to the Stabilizer

        :param qubit_position: the qubit position where the gate is applied
        :type qubit_position: int
        :return: nothing
        :rtype: None
        """
        self._mixture = [
            (pi, transform.z_gate(tableau_i, qubit_position))
            for (pi, tableau_i) in self._mixture
        ]

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
        self._mixture = [
            (pi, sfc.reset_z(tableau_i, qubit_position, 0, measurement_determinism))
            for (pi, tableau_i) in self._mixture
        ]

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
        raise NotImplementedError
        # self._tableau = sfc.remove_qubit(
        #     self._tableau, qubit_position, measurement_determinism
        # )

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
        raise NotImplementedError

        # self._tableau = sfc.partial_trace(
        #     self._tableau,
        #     keep=qubit_positions,
        #     dims=self.n_qubit * [2],
        #     measurement_determinism=measurement_determinism,
        # )

    def __str__(self):
        """
        Return a string representation of this state representation

        :return: a string representation of this state representation
        :rtype: str
        """
        return "".join(f"{pi}: {ti.__str__()}\n\n" for (pi, ti) in self._mixture)

    def __eq__(self, other):
        """
        Compare two Stabilizer objects

        :param other: the other Stabilizer to be compared
        :type other: Stabilizer
        :return: True if the stabilizer tableaux of two Stabilizer objects are the same
        :rtype: bool
        """
        raise NotImplementedError
