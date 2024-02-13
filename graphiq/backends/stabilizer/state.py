"""
State representation using the stabilizer formalism
"""

import numpy as np

import graphiq.backends.stabilizer.functions.clifford as sfc
import graphiq.backends.stabilizer.functions.transformation as transform
from graphiq.backends.stabilizer.clifford_tableau import CliffordTableau
from graphiq.backends.stabilizer.functions.stabilizer import canonical_form
from graphiq.backends.state_base import StateRepresentationBase


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
    def n_qubits(self):
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
        (
            self._tableau,
            outcome,
            _,
        ) = sfc.z_measurement_gate(
            self._tableau, qubit_position, measurement_determinism
        )
        return outcome

    def apply_x_measurement(
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
        (
            self._tableau,
            outcome,
            _,
        ) = sfc.x_measurement_gate(
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

    def apply_phase_dagger(self, qubit_position):
        """
        Apply the phase dagger gate to the Stabilizer

        :param qubit_position: the qubit position where the gate is applied
        :type qubit_position: int
        :return: nothing
        :rtype: None
        """
        self._tableau = transform.phase_dagger_gate(self._tableau, qubit_position)

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
            dims=self.n_qubits * [2],
            measurement_determinism=measurement_determinism,
        )

    def partial_trace(self, keep, dims):
        """
        Trace out qubits after disentangling them from the rest

        :param keep:
        :type keep: list[int] or numpy.ndarray
        :param dims:
        :type dims:
        :return: nothing
        :rtype: None
        """
        self._tableau = sfc.partial_trace(self._tableau, keep=keep, dims=dims)

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


class MixedStabilizer(StateRepresentationBase):
    """
    A mixed state representation using the stabilizer formalism, where the mixture is represented as a list of
    pure states (tableaus) and an associated mixture probability.
    """

    def __init__(self, data, *args, **kwargs):
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
                isinstance(p_i, float) and isinstance(t_i, CliffordTableau)
                for (p_i, t_i) in data
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
                isinstance(p_i, float) and isinstance(t_i, CliffordTableau)
                for (p_i, t_i) in data
            )
        return valid

    @property
    def n_qubits(self):
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
        """
        return self._mixture

    @mixture.setter
    def mixture(self, value):
        """
        Sets the mixture of pure states, represented as a list of tableaus and associated probabilities.

        :param value: a new mixture list, pure tableau, or a parameter to initialize a new tableau
        :type value: list or int or CliffordTableau
        :return: the mixture as a list of (probability_i, tableau_i)
        :rtype: list
        """
        if isinstance(value, list):
            assert all(
                isinstance(p_i, float) and isinstance(t_i, CliffordTableau)
                for (p_i, t_i) in value
            )
            assert (
                len(set([t_i.n_qubits for p_i, t_i in value])) == 1
            )  # all tableaux are same number of qubits
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
        The data that represents the state given by the MixedStabilizer representation

        :return: the mixture that represents this state
        :rtype: list
        """
        return self.mixture

    @data.setter
    def data(self, value):
        """
        Set the data that represents the state given by the MixedStabilizer representation

        :param value: a new tableau or a parameter to initialize a new tableau
        :type value: CliffordTableau or int
        :return: nothing
        :rtype: None
        """
        self.mixture = value

    @property
    def probability(self):
        r"""
        Computes the total probability as the summed probability of all pure states in the mixture
        $\sum_i p_i \\ \forall (p_i, \mathcal{T}_i)$.

        :return: sum of probabilities
        :rtype: float
        """
        return sum(p_i for p_i, t_i in self.mixture)

    @property
    def tableau(self):
        return TypeError(
            "Simulating using a mixed state representation, no tableau defined."
        )

    def reduce(self):
        """
        Reduce the number of tableaux store in the mixture by comparing the Hamming distance between them.
        Probabilities are summed and one tableau removed if they are the same.

        :return: nothing
        :rtype: None
        """
        # TODO: explore other ways of reduction and further simplification using a standard form.
        mixture_temp = self._mixture
        mixture_reduce = []
        while len(mixture_temp) != 0:
            p0, t0 = mixture_temp[0]
            mixture_temp.pop(0)
            for i, (p_i, t_i) in enumerate(mixture_temp):
                if np.count_nonzero(t0 != t_i) == 0:
                    p0 += p_i
                    mixture_temp.pop(i)

            mixture_reduce.append((p0, t0))
        self._mixture = mixture_reduce

    def apply_unitary(self, qubit_position, unitary):
        raise NotImplementedError(
            "Stabilizer backend does not support general unitary operation."
        )

    def apply_conditioned_gate(self, qubit_position, outcomes, gate=None):
        """
        Apply a single-qubit gate, conditioned on a classical measurement outcome.

        :param qubit_position: int
        :param outcomes: list of measurement outcomes for each tableau in the mixture
        :param gate: str, one of 'x', 'y', 'z', or 'h'
        :return:
        """
        assert isinstance(outcomes, list)
        assert len(outcomes) == len(self._mixture)

        if gate == "x":
            trans = transform.x_gate
        elif gate == "y":
            trans = transform.y_gate
        elif gate == "z":
            trans = transform.z_gate
        elif gate == "h":
            trans = transform.hadamard_gate
        else:
            raise NotImplementedError(
                "Gate must be provided for conditioning measurement outcomes."
            )
        for i, outcome in enumerate(outcomes):
            if outcome == 1:
                p_i, t_i = self._mixture[i]
                self._mixture[i] = (p_i, trans(t_i, qubit_position))

    def apply_measurement(
        self, qubit_position, measurement_determinism="probabilistic"
    ):
        """
        Apply the measurement in the computational basis to a given qubit. For the MixedStabilizer state,
        we measure the outcome for each tableau in the mixture, returning a list of outcomes.

        # todo think of classical probabilities being stored on the c-registers?

        :param qubit_position: the qubit position where the measurement is applied
        :type qubit_position: int
        :param measurement_determinism: if "probabilistic", measurement results are probabilistically selected
                if 1, measurement results default to 1 unless the probability of measuring p(1) = 0
                if 0, measurement results default to 0 unless the probability of measuring p(0) = 0
        :type measurement_determinism: str/int
        :return: the measurement outcome
        :rtype: list
        """
        outcomes = []
        for i, (p_i, t_i) in enumerate(self._mixture):
            tableau, outcome, x_p = sfc.z_measurement_gate(
                t_i, qubit_position, measurement_determinism=measurement_determinism
            )
            outcomes.append(outcome)
            self._mixture[i] = (p_i, tableau)

        return outcomes

    def apply_hadamard(self, qubit_position):
        """
        Apply the Hadamard gate to the Stabilizer

        :param qubit_position: the qubit position where the gate is applied
        :type qubit_position: int
        :return: nothing
        :rtype: None
        """
        self._mixture = [
            (p_i, transform.hadamard_gate(t_i, qubit_position))
            for (p_i, t_i) in self._mixture
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
            (p_i, transform.cnot_gate(t_i, control, target))
            for (p_i, t_i) in self._mixture
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
            (p_i, transform.control_z_gate(t_i, control, target))
            for (p_i, t_i) in self._mixture
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
            (p_i, transform.phase_gate(t_i, qubit_position))
            for (p_i, t_i) in self._mixture
        ]

    def apply_phase_dagger(self, qubit_position):
        """
        Apply the phase dagger gate to the Stabilizer

        :param qubit_position: the qubit position where the gate is applied
        :type qubit_position: int
        :return: nothing
        :rtype: None
        """
        self._mixture = [
            (p_i, transform.phase_dagger_gate(t_i, qubit_position))
            for (p_i, t_i) in self._mixture
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
            (p_i, transform.x_gate(t_i, qubit_position)) for (p_i, t_i) in self._mixture
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
            (p_i, transform.y_gate(t_i, qubit_position)) for (p_i, t_i) in self._mixture
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
            (p_i, transform.z_gate(t_i, qubit_position)) for (p_i, t_i) in self._mixture
        ]

    def reset_qubit(self, qubit_position, measurement_determinism="probabilistic"):
        r"""
        Reset a given qubit to $|0\rangle$ state after disentangling it from the rest

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
            (p_i, sfc.reset_z(t_i, qubit_position, 0, measurement_determinism))
            for (p_i, t_i) in self._mixture
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
        self._mixture = [
            (p_i, sfc.remove_qubit(t_i, qubit_position, measurement_determinism))
            for (p_i, t_i) in self._mixture
        ]

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
        self._mixture = [
            (
                p_i,
                sfc.partial_trace(
                    t_i,
                    keep=qubit_positions,
                    dims=self.n_qubits * [2],
                    measurement_determinism=measurement_determinism,
                ),
            )
            for (p_i, t_i) in self._mixture
        ]

    def partial_trace(self, keep, dims):
        """
        Trace out qubits after disentangling them from the rest

        :param keep: the qubit positions to be kept
        :type keep: list[int] or numpy.ndarray
        :param dims: dimension of each subsystem
        :type dims: list[int] or numpy.ndarray
        :return: nothing
        :rtype: None
        """
        self._mixture = [
            (
                p_i,
                sfc.partial_trace(
                    t_i,
                    keep=keep,
                    dims=dims,
                ),
            )
            for (p_i, t_i) in self._mixture
        ]

    def sort(self):
        """
        Sort the mixture according to descending order of probabilities

        :return: nothing
        :rtype: None
        """
        self.mixture = sorted(self._mixture, key=lambda item: item[0], reverse=True)

    def __str__(self):
        """
        Return a string representation of this state representation

        :return: a string representation of this state representation
        :rtype: str
        """
        s = f"{self.__class__.__name__} | {len(self._mixture)} tableaux in mixture"
        return s

    def __eq__(self, other):
        """
        Compare two MixedStabilizer objects

        :param other: the other MixedStabilizer to be compared
        :type other: MixedStabilizer
        :return: True if the stabilizer tableaux of two MixedStabilizer objects are the same
        :rtype: bool
        """
        # Treat two objects the same if and only if they have the same probability distribution
        # and same set of tableaux with the same probability
        if len(self._mixture) != len(other.mixture):
            return False
        # sort first
        self.sort()
        other.sort()
        for i in range(len(self._mixture)):
            if not np.isclose(self._mixture[i][0], other.mixture[i][0]):
                return False
            else:
                if self._mixture[i][1] != other.mixture[i][1]:
                    return False

        return True
