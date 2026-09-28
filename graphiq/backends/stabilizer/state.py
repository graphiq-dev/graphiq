# Copyright (c) 2022-2024 Quantum Bridge Technologies Inc.
# Copyright (c) 2022-2024 Ki3 Photonics Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
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
    """
    A pure state representation using the stabilizer formalism, backed by a single
    CliffordTableau.

    General unitaries are not supported; the state evolves by Clifford gates, Z and X
    measurements, reset to |0>, and the qubit-removing operations (remove, trace out). For a
    probability-weighted mixture of tableaus, see MixedStabilizer.
    """

    def __init__(self, data, *args, **kwargs):
        """
        Create a Stabilizer representation object

        :param data: the number of qubits to initialize in the all-zero state, or an existing
            tableau to adopt (not copied)
        :type data: int or CliffordTableau
        :raises TypeError: if data is neither an int nor a CliffordTableau
        :return: nothing
        :rtype: None
        """
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
        """
        Whether data is an accepted input for this representation

        :param data: the candidate data
        :type data: any
        :return: True if data is an int or a CliffordTableau
        :rtype: bool
        """
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
        """
        Not supported: the stabilizer formalism represents Clifford operations only.

        :param qubit_position: the qubit the unitary would act on
        :type qubit_position: int
        :param unitary: the unitary that would be applied
        :type unitary: numpy.ndarray
        :raises NotImplementedError: always
        :return: nothing
        :rtype: None
        """
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
        Apply the measurement in the X basis to a given qubit

        :param qubit_position: the qubit position where the measurement is applied
        :type qubit_position: int
        :param measurement_determinism: if "probabilistic", measurement results are probabilistically selected
                if 1, measurement results default to 1 unless the probability of measuring p(1) = 0
                if 0, measurement results default to 0 unless the probability of measuring p(0) = 0
        :type measurement_determinism: str/int
        :return: the measurement outcome
        :rtype: int
        """
        outcome = sfc.measure_x(self._tableau, qubit_position, measurement_determinism)
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
        # partial_trace takes the qubits to keep, not the ones to trace out
        keep = [q for q in range(self.n_qubits) if q not in set(qubit_positions)]
        self._tableau = sfc.partial_trace(
            self._tableau,
            keep=keep,
            dims=self.n_qubits * [2],
            measurement_determinism=measurement_determinism,
        )

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
        :return: True if the stabilizer tableaux of two Stabilizer objects are
            the same; NotImplemented if other is not a Stabilizer
        :rtype: bool or NotImplementedType
        """
        if not isinstance(other, Stabilizer):
            return NotImplemented
        tableau1 = canonical_form(self.data.to_stabilizer())
        tableau2 = canonical_form(other.data.to_stabilizer())
        return tableau1 == tableau2


class MixedStabilizer(StateRepresentationBase):
    """
    A mixed state representation using the stabilizer formalism, where the mixture is represented as a list of
    pure states (tableaus) and an associated mixture probability.
    """

    def __init__(self, data, *args, **kwargs):
        """
        Create a MixedStabilizer representation object

        An int or a single tableau becomes a one-element mixture of probability 1.0.

        :param data: the number of qubits to initialize in the all-zero state, a single tableau,
            or a ready mixture as a list of (probability_i, tableau_i)
        :type data: int or CliffordTableau or list[tuple]
        :raises TypeError: if data is none of those
        :raises AssertionError: if a list entry is not a (float, CliffordTableau) pair
        :return: nothing
        :rtype: None
        """
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
        """
        Whether data is an accepted input for this representation

        A list is accepted only if every entry is a (float, CliffordTableau) pair.

        :param data: the candidate data
        :type data: any
        :return: True if data is an int, a CliffordTableau, or such a list
        :rtype: bool
        """
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
        :raises AssertionError: if a list entry is not a (float, CliffordTableau) pair, or if the tableaux
            do not all have the same number of qubits
        :raises TypeError: if value is not a list, a CliffordTableau or an int
        :return: nothing
        :rtype: None
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

        :param value: a new mixture list, pure tableau, or a parameter to initialize a new tableau
        :type value: list or CliffordTableau or int
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
        """
        A mixture has one tableau per branch, not one overall; read
        :attr:`mixture` instead.

        :raises TypeError: always
        """
        raise TypeError(
            "Simulating using a mixed state representation, no tableau defined."
        )

    @staticmethod
    def _state_key(tableau):
        """
        A hashable key identifying the STATE a branch's tableau represents,
        as opposed to the raw tableau data.

        Two CliffordTableau that represent the same state can differ in their
        destabilizer rows (a free choice of completion) and in the order/linear
        combination of their stabilizer generators, so raw data equality
        (``CliffordTableau.__eq__``) is too strict for grouping branches of a
        mixture. The key is instead built from the stabilizer half only, put
        into canonical form, together with its phase -- phase is included
        because two branches whose stabilizer groups agree up to canonical
        form but differ in sign (e.g. |0> vs |1>) are physically distinct
        states and must not be folded together.

        ``canonical_form`` mutates the StabilizerTableau it is given in place,
        and both halves of that tableau are views into the CliffordTableau
        until ``StabilizerTableau.__init__`` copies them:
        ``CliffordTableau.stabilizer`` is a view into the table, and
        ``to_stabilizer`` slices the phase vector rather than copying it. Two
        separate copies therefore stand between this and the branch being
        reduced -- ``np.copy`` for the table, and ``phase.astype(int)`` for the
        phase. The phase one is the easier to lose in a refactor and the more
        damaging: ``tab_row_swap`` assigns ``tableau.phase = row_swap(...)``,
        and ``row_swap`` mutates its argument in place before the setter runs,
        so without that copy an ordinary canonical-form row swap would corrupt
        the parent CliffordTableau's phase vector directly.

        Internally, the row operations ``canonical_form`` performs discard
        the i-phase of each row (only the +-1 phase is tracked). This is
        harmless specifically inside canonical form reduction of a
        stabilizer's own generators: since stabilizer generators mutually
        commute, the sum of any two rows' g-function terms is always even, so
        no residual factor of i is ever dropped on the floor.

        Separately from that, the key ignores the CliffordTableau's whole
        i-phase vector, destabilizer and stabilizer rows alike, because
        ``to_stabilizer`` keeps only the +-1 phase and a StabilizerTableau has
        no i-phase at all. ``reduce`` does not check the i-phase either, so a
        branch with a non-zero i-phase is folded into the first branch of the
        same state and disappears; the i-phase check made when converting to
        a density matrix sees it only if it is the branch kept. No
        correct gate produces such a tableau.

        :param tableau: a branch's tableau
        :type tableau: CliffordTableau
        :return: a hashable key identifying the state
        :rtype: tuple[bytes, bytes]
        """
        canonical = canonical_form(tableau.to_stabilizer())
        table_bytes = np.ascontiguousarray(canonical.table.astype(np.int8)).tobytes()
        phase_bytes = np.ascontiguousarray(canonical.phase.astype(np.int8)).tobytes()
        return table_bytes, phase_bytes

    def reduce(self):
        """
        Fold branches that represent the same state into a single entry,
        summing their probabilities.

        Branches are grouped by state identity (see ``_state_key``) rather
        than by raw tableau equality, so two branches that reach the same
        state by different routes -- e.g. different destabilizer completions,
        or stabilizer generators produced in a different order -- fold
        together even though ``CliffordTableau.__eq__`` would call them
        unequal.

        ``reduce`` itself costs more per call than a raw-equality scan, since
        ``canonical_form`` is O(n^3) Python-level row operations paid once per
        branch. The simulation as a whole is faster, because it merges
        branches that raw equality misses (on a noisy compile at
        n_emitters=3, a mixture of 64 branches reduces to 8), and every later
        operation on the mixture iterates all of it.

        :return: nothing
        :rtype: None
        """
        # Grouping builds a new list rather than mutating in place: the
        # mixture setter stores the caller's list by reference, so rewriting
        # it here would change it out from under the caller. `order` records
        # the first-seen order of the groups, which the result follows.
        groups = {}
        order = []
        for p_i, t_i in self._mixture:
            key = self._state_key(t_i)
            if key in groups:
                p0, t0 = groups[key]
                groups[key] = (p0 + p_i, t0)
            else:
                groups[key] = (p_i, t_i)
                order.append(key)
        self._mixture = [groups[key] for key in order]

    def apply_unitary(self, qubit_position, unitary):
        """
        Not supported: the stabilizer formalism represents Clifford operations only.

        :param qubit_position: the qubit the unitary would act on
        :type qubit_position: int
        :param unitary: the unitary that would be applied
        :type unitary: numpy.ndarray
        :raises NotImplementedError: always
        :return: nothing
        :rtype: None
        """
        raise NotImplementedError(
            "Stabilizer backend does not support general unitary operation."
        )

    def apply_conditioned_gate(self, qubit_position, outcomes, gate=None):
        """
        Apply a single-qubit gate, conditioned on a classical measurement outcome.

        :param qubit_position: the qubit position where the gate is applied
        :type qubit_position: int
        :param outcomes: list of measurement outcomes for each tableau in the mixture; the gate is applied to
            the tableaux whose outcome is 1
        :type outcomes: list[int]
        :param gate: one of 'x', 'y', 'z', or 'h'
        :type gate: str
        :raises AssertionError: if outcomes is not a list with one entry per tableau in the mixture
        :raises NotImplementedError: if gate is not one of 'x', 'y', 'z', or 'h'
        :return: nothing
        :rtype: None
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
        # partial_trace takes the qubits to keep, not the ones to trace out.
        # keep is computed once, before the comprehension, for every term.
        n_qubits = self.n_qubits
        keep = [q for q in range(n_qubits) if q not in set(qubit_positions)]
        self._mixture = [
            (
                p_i,
                sfc.partial_trace(
                    t_i,
                    keep=keep,
                    dims=n_qubits * [2],
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
        :return: True if the two mixtures have the same length and, after each is sorted by
            probability, every pair of entries has the same probability (up to np.isclose) and
            the same stabilizer tableau; NotImplemented if other is not a MixedStabilizer
        :rtype: bool or NotImplementedType
        """
        if not isinstance(other, MixedStabilizer):
            return NotImplemented
        # Treat two objects the same if and only if they have the same probability distribution
        # and same set of tableaux with the same probability
        if len(self._mixture) != len(other.mixture):
            return False
        # Compare sorted copies so the comparison does not reorder either
        # object's mixture as a side effect.
        mixture1 = sorted(self._mixture, key=lambda item: item[0], reverse=True)
        mixture2 = sorted(other.mixture, key=lambda item: item[0], reverse=True)
        for i in range(len(mixture1)):
            if not np.isclose(mixture1[i][0], mixture2[i][0]):
                return False
            else:
                if mixture1[i][1] != mixture2[i][1]:
                    return False

        return True
