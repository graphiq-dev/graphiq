r"""
Deterministic solver which follows the paper by Li et al [1]_.

.. [1] Bikun Li, Sophia E. Economou and Edwin Barnes, npj. Quantum Information 8, 11 (2022)

"""
import numpy as np

import src.backends.stabilizer.functions.height as height
import src.backends.stabilizer.functions.stabilizer as sfs
import src.backends.stabilizer.functions.transformation as transform
from src.backends.stabilizer.tableau import StabilizerTableau
from src.backends.compiler_base import CompilerBase

import src.noise.noise_models as nm
from src.metrics import MetricBase
from src.solvers import SolverBase
from src.circuit import CircuitDAG
from src.io import IO
from src import ops


class DeterministicSolver(SolverBase):
    """
    This deterministic solver finds a quantum circuit that produces the target state in a time-reversed fashion.
    """

    name = "deterministic"

    one_qubit_ops = list(ops.one_qubit_cliffords())

    def __init__(
        self,
        target,
        metric: MetricBase,
        compiler: CompilerBase,
        circuit: CircuitDAG = None,
        io: IO = None,
        noise_model_mapping=None,
        *args,
        **kwargs,
    ):
        """
        Initialize a DeterministicSolver object

        :param target: target quantum state
        :type target: QuantumState
        :param metric: a metric to be evaluated for the final state and circuit
        :type metric: MetricBase
        :param compiler: a backend compiler to be used for compilation of the quantum circuit
        :type compiler: CompilerBase
        :param circuit: an initial quantum circuit
        :type circuit: CircuitDAG
        :param io: input/output object for saving logs, intermediate results, circuits, etc.
        :type io: IO
        :param noise_model_mapping: a dictionary that associates each operation type to a noise model
        :type noise_model_mapping: dict
        """

        super().__init__(target, metric, compiler, circuit, io, *args, **kwargs)

        tableau = target.stabilizer.data.to_stabilizer()
        self.n_emitter = self.determine_n_emitters(tableau)
        self.n_photon = tableau.n_qubits
        self.noise_simulation = True

        if noise_model_mapping is None:
            noise_model_mapping = {}
            self.noise_simulation = False
        elif type(noise_model_mapping) is not dict:
            raise TypeError(
                f"Datatype {type(noise_model_mapping)} is not a valid noise_model_mapping. "
                f"noise_model_mapping should be a dict or None"
            )
        self.noise_model_mapping = noise_model_mapping

    def solve(self):
        """
        The main function for the solver

        :return: nothing
        :rtype: None
        """

        self.compiler.noise_simulation = self.noise_simulation
        # initialize the circuit
        circuit = CircuitDAG(
            n_emitter=self.n_emitter, n_photon=self.n_photon, n_classical=1
        )

        stabilizer_tableau = self.target.stabilizer.data.to_stabilizer()

        # add emitter qubits in |0> state
        for _ in range(self.n_emitter):
            stabilizer_tableau = sfs.insert_qubit(
                stabilizer_tableau, stabilizer_tableau.n_qubits
            )

        # main loop
        emitter_depth = np.zeros(self.n_emitter)
        for j in range(self.n_photon, 0, -1):
            # transform the stabilizers into echelon gauge
            stabilizer_tableau = sfs.rref(stabilizer_tableau)
            height_list = height.height_func_list(
                stabilizer_tableau.x_matrix, stabilizer_tableau.z_matrix
            )

            height_list = [0] + height_list
            if height_list[j] < height_list[j - 1]:
                # apply time-reversed measurement and update the tableau
                self._time_reversed_measurement(
                    circuit, emitter_depth, stabilizer_tableau, j - 1
                )
                stabilizer_tableau = sfs.rref(stabilizer_tableau)

            # apply photon-absorption and update the stabilizer tableau
            self._add_photon_absorption(
                circuit, emitter_depth, stabilizer_tableau, j - 1
            )

        stabilizer_tableau = sfs.rref(stabilizer_tableau)

        # make sure that all photonic qubits are in the |0> state
        assert np.array_equal(
            stabilizer_tableau.x_matrix[0 : self.n_photon, 0 : self.n_photon],
            np.zeros((self.n_photon, self.n_photon)),
        )
        assert np.array_equal(
            stabilizer_tableau.z_matrix[0 : self.n_photon, 0 : self.n_photon],
            np.eye(self.n_photon),
        )

        # transform all the emitter qubits to the |0> state
        _, inverse_circuit = sfs.inverse_circuit(stabilizer_tableau.copy())

        self._add_gates_from_str(
            circuit, emitter_depth, stabilizer_tableau, inverse_circuit
        )

        # correct the phase of generators that act only on emitter qubits
        for i in range(self.n_emitter):
            if stabilizer_tableau.phase[self.n_photon + i] == 1:
                transform.x_gate(stabilizer_tableau, self.n_photon + i)
                self._add_one_qubit_gate(
                    circuit, emitter_depth, [ops.SigmaX], self.n_photon + i
                )

        # final circuit
        circuit.validate()

        compiled_state = self.compiler.compile(circuit)
        compiled_state.partial_trace(
            keep=[*range(self.n_photon)], dims=(self.n_photon + self.n_emitter) * [2]
        )
        # evaluate the metric
        score = self.metric.evaluate(compiled_state, circuit)
        self.result = (score, circuit.copy())

    @staticmethod
    def determine_n_emitters(tableau):
        """
        Determine the minimum number of emitters needed

        :param tableau: a tableau that represents the stabilizer state
        :type tableau: StabilizerTableau
        :return: the minimum number of emitters required to generate the state
        :rtype: int
        """
        tableau = sfs.rref(tableau)
        height_list = height.height_func_list(tableau.x_matrix, tableau.z_matrix)
        return max(height_list)

    def _time_reversed_measurement(self, circuit, emitter_depth, tableau, photon_index):
        """
        Apply the time-reversed measurement

        :param circuit: a quantum circuit
        :type circuit: CircuitDAG
        :param emitter_depth: a list of circuit depths for emitter registers
        :type emitter_depth: np.ndarray
        :param tableau: a stabilizer tableau
        :type tableau: StabilizerTableau
        :param photon_index: index of the photon for the conditional X gate
        :type photon_index: int
        :return: nothing
        :rtype: None
        """
        # find the generators that act only on the emitters
        sum_table = (
            tableau.x_matrix[:, 0 : self.n_photon]
            + tableau.z_matrix[:, 0 : self.n_photon]
        )
        possible_generators = np.where(~sum_table.any(axis=1))[0]

        assert len(possible_generators) > 0
        # In Li et al.'s implementation, it always picks the first one
        generator_index = possible_generators[0]

        # find the first nontrivial emitter position in this generator
        emitter_indices = self._find_emitter_indices(tableau, generator_index)
        emitter_index = int(emitter_indices[0])

        self._single_out_emitter(
            circuit, emitter_depth, tableau, generator_index, emitter_index
        )

        transform.hadamard_gate(tableau, self.n_photon + emitter_index)
        self._add_measurement_cnot_and_reset(circuit, emitter_index, photon_index)

        # transform this generator so that it acts nontrivially on only one emitter and that Pauli is X
        emitter_depth[emitter_index] += 1
        transform.cnot_gate(tableau, self.n_photon + emitter_index, photon_index)

    def _add_photon_absorption(self, circuit, emitter_depth, tableau, photon_index):
        """
        Add a photon absorption (time-reversed photon emission) event

        :param circuit: a quantum circuit
        :type circuit: CircuitDAG
        :param emitter_depth: a list of circuit depths for emitter registers
        :type emitter_depth: numpy.ndarray
        :param tableau: a stabilizer tableau
        :type tableau: StabilizerTableau
        :param photon_index: index of photon to be absorbed
        :type photon_index: int
        :return: nothing
        :rtype: None
        """
        for i in range(tableau.n_qubits - 1, -1, -1):
            if height.leftmost_nontrivial_index(tableau, i) == photon_index:
                generator_index = i
                break

        gate_list = self._change_pauli_type(tableau, generator_index, photon_index, "z")
        self._add_one_qubit_gate(circuit, emitter_depth, gate_list, photon_index)

        emitter_indices = self._find_emitter_indices(tableau, generator_index)
        emitter_index = int(emitter_indices[0])
        for i in range(self.n_emitter):
            # transform this generator so that it has only Z on emitter registers
            gate_list = self._change_pauli_type(
                tableau,
                generator_index,
                self.n_photon + i,
                "z",
            )
            self._add_one_qubit_gate(
                circuit, emitter_depth, gate_list, self.n_photon + i
            )

        # now this generator contains only ZZ on emitter qubits
        # transform this generator so that for emitter qubits, there is only one Z.
        self._transform_generator_emitters(
            circuit, emitter_depth, tableau, generator_index, emitter_index, "z"
        )

        # if the phase of this generator is -1, flip it by adding X gate to the emitter qubit
        if tableau.phase[generator_index] == 1:
            transform.x_gate(tableau, self.n_photon + emitter_index)
            self._add_one_qubit_gate(
                circuit, emitter_depth, [ops.SigmaX], self.n_photon + emitter_index
            )

        # emit (absorb) this photon by the chosen emitter
        self._add_emitter_photon_cnot(
            circuit, emitter_depth, emitter_index, photon_index
        )
        transform.cnot_gate(tableau, self.n_photon + emitter_index, photon_index)

        # transform all generators (except the chosen generator) with Z on this photonic register
        # by eliminating Z
        pauli_z_list = sfs.one_pauli_type_finder(
            tableau.x_matrix, tableau.z_matrix, [0, photon_index], "z"
        )

        original_row = np.copy(tableau.table[generator_index])
        original_phase = tableau.phase[generator_index]
        pauli_z_list = np.setdiff1d(pauli_z_list, [generator_index])
        for i in pauli_z_list:

            # right multiplication needed where rowsum is left multiplication
            sfs.tab_row_swap(tableau, generator_index, i)
            sfs.tab_row_sum(tableau, generator_index, i)
            tableau.table[generator_index] = original_row
            tableau.phase[generator_index] = original_phase

    def _add_gates_from_str(self, circuit, emitter_depth, tableau, gate_str_list):
        """
        Add gates to disentangle all emitter qubits. This is used in the last step.

        :param circuit: a quantum circuit
        :type circuit: CircuitDAG
        :param emitter_depth: a list of circuit depths for emitter registers
        :type emitter_depth: numpy.ndarray
        :param tableau: a stabilizer tableau
        :type tableau: StabilizerTableau
        :param gate_str_list: a list of gates to be applied
        :type gate_str_list: list[(str, int) or (str, int, int)]
        :return: nothing
        :rtype: None
        """
        for gate in gate_str_list:
            if gate[0] == "H":
                self._add_one_qubit_gate(
                    circuit, emitter_depth, [ops.Hadamard], gate[1]
                )
                transform.hadamard_gate(tableau, gate[1])
            elif gate[0] == "P":
                # add the inverse of the phase gate
                self._add_one_qubit_gate(
                    circuit, emitter_depth, [ops.SigmaZ, ops.Phase], gate[1]
                )
                transform.phase_gate(tableau, gate[1])
            elif gate[0] == "X":
                self._add_one_qubit_gate(circuit, emitter_depth, [ops.SigmaX], gate[1])
                transform.x_gate(tableau, gate[1])
            elif gate[0] == "CNOT":
                self._add_one_emitter_cnot(
                    circuit,
                    emitter_depth,
                    gate[1] - self.n_photon,
                    gate[2] - self.n_photon,
                )
                transform.cnot_gate(tableau, gate[1], gate[2])
            elif gate[0] == "CZ":
                self._add_one_qubit_gate(
                    circuit, emitter_depth, [ops.Hadamard], gate[2]
                )
                transform.hadamard_gate(tableau, gate[2])
                self._add_one_emitter_cnot(
                    circuit,
                    emitter_depth,
                    gate[1] - self.n_photon,
                    gate[2] - self.n_photon,
                )
                transform.cnot_gate(tableau, gate[1], gate[2])
                self._add_one_qubit_gate(
                    circuit, emitter_depth, [ops.Hadamard], gate[2]
                )
                transform.hadamard_gate(tableau, gate[2])
            else:
                raise ValueError("Invalid gate in the list.")

    def _change_pauli_type(self, tableau, row, column, result="z"):
        """
        Transform the Pauli of the given qubit (column) in the given generator (row)
        so that it becomes the Pauli that is specified by the input parameter (result).

        :param tableau: a stabilizer tableau of the state
        :type tableau: StabilizerTableau
        :param row: the generator index in the tableau
        :type row: int
        :param column: the qubit index in the tableau
        :type column: int
        :param result: the Pauli type to be transformed to
        :type result: str
        :return: a list of one-qubit gates to be added to the circuit
        :rtype: list[ops.OperationBase]
        """
        gate_list = []
        if tableau.x_matrix[row, column] == 1 and tableau.z_matrix[row, column] == 0:
            # current is X
            if result.lower() == "z":
                transform.hadamard_gate(tableau, column)
                gate_list.append(ops.Hadamard)
            elif result.lower() == "y":
                transform.phase_gate(tableau, column)
                # add the inverse of phase gate to the gate list
                gate_list.append(ops.SigmaZ)
                gate_list.append(ops.Phase)

        elif tableau.x_matrix[row, column] == 1 and tableau.z_matrix[row, column] == 1:
            # current is Y
            if result.lower() == "z":
                transform.phase_dagger_gate(tableau, column)
                transform.hadamard_gate(tableau, column)

                # add the inverse of phase dagger gate and inverse of Hadamard gate to the gate list
                gate_list.append(ops.Phase)
                gate_list.append(ops.Hadamard)
            elif result.lower() == "x":
                transform.phase_dagger_gate(tableau, column)

                # add the inverse of phase dagger gate to the gate list
                gate_list.append(ops.Phase)

        if tableau.x_matrix[row, column] == 0 and tableau.z_matrix[row, column] == 1:
            # current is Z
            if result.lower() == "x":
                transform.hadamard_gate(tableau, column)
                gate_list.append(ops.Hadamard)
            elif result.lower() == "y":
                transform.hadamard_gate(tableau, column)
                transform.phase_gate(tableau, column)

                # add the inverse of Hadamard gate and the inverse of phase gate to the gate list
                gate_list.append(ops.Hadamard)
                gate_list.append(ops.SigmaZ)
                gate_list.append(ops.Phase)

        return gate_list

    def _add_one_qubit_gate(self, circuit, emitter_depth, gate_list, index):
        """
        Add a one-qubit gate to the circuit

        :param circuit: a quantum circuit
        :type circuit: CircuitDAG
        :param emitter_depth: a list of circuit depths for emitter registers
        :type emitter_depth: numpy.ndarray
        :param gate_list: a list of one-qubit gates to be added to the circuit
        :type gate_list: list[ops.OperationBase]
        :param index: the qubit position where this one-qubit gate is applied
        :type index: int
        :return: nothing
        :rtype: None
        """

        if index >= self.n_photon:
            reg_type = "e"
            reg = index - self.n_photon
        else:
            reg_type = "p"
            reg = index

        edge = circuit.dag.out_edges(nbunch=f"{reg_type}{reg}_in", keys=True)
        edge = list(edge)[0]
        next_node = circuit.dag.nodes[edge[1]]
        next_op = next_node["op"]
        if isinstance(next_op, ops.OneQubitGateWrapper):
            # if two OneQubitGateWrapper gates are next to each other, combine them
            gate_list = next_op.operations + gate_list
            gate_list = ops.simplify_local_clifford(gate_list)
            if gate_list == [ops.Identity, ops.Identity]:
                circuit.remove_op(edge[1])
                if reg_type == "e":
                    emitter_depth[reg] -= 1
                return
        else:
            # simplify the gate to be one of the 24 local Clifford gates
            gate_list = ops.simplify_local_clifford(gate_list)
            if gate_list == [ops.Identity, ops.Identity]:
                return
        noise = self._wrap_noise(gate_list, self.noise_model_mapping)
        gate = ops.OneQubitGateWrapper(
            gate_list,
            reg_type=reg_type,
            register=reg,
            noise=noise,
        )
        if isinstance(next_op, ops.OneQubitGateWrapper):
            circuit.replace_op(edge[1], gate)
        else:
            circuit.insert_at(gate, [edge])
            if reg_type == "e":
                emitter_depth[reg] += 1

    def _add_one_emitter_cnot(
        self, circuit, emitter_depth, control_emitter, target_emitter
    ):
        """
        Add a CNOT between two emitter qubits

        :param circuit: a quantum circuit
        :type circuit: CircuitDAG
        :param emitter_depth: a list of circuit depths for emitter registers
        :type emitter_depth: numpy.ndarray
        :param control_emitter: register index of the control emitter
        :type control_emitter: int
        :param target_emitter: register index of the target emitter
        :type target_emitter: int
        :return: nothing
        :rtype: None
        """

        control_edge = circuit.dag.out_edges(nbunch=f"e{control_emitter}_in", keys=True)
        edge0 = list(control_edge)[0]
        target_edge = circuit.dag.out_edges(nbunch=f"e{target_emitter}_in", keys=True)
        edge1 = list(target_edge)[0]
        gate = ops.CNOT(
            control=circuit.dag.edges[edge0]["reg"],
            control_type="e",
            target=circuit.dag.edges[edge1]["reg"],
            target_type="e",
            noise=self._identify_noise(ops.CNOT.__name__, self.noise_model_mapping),
        )
        circuit.insert_at(gate, [edge0, edge1])
        emitter_depth[control_emitter] += 1
        emitter_depth[target_emitter] += 1

    def _transform_generator_emitters(
        self, circuit, emitter_depth, tableau, generator_index, emitter_index, result
    ):
        """
        Transform a given generator such that it contains only one type of Pauli (specified by the parameter result)
        on all emitter qubits

        :param circuit: a quantum circuit
        :type circuit: CircuitDAG
        :param emitter_depth: a list of circuit depths for emitter registers
        :type emitter_depth: numpy.ndarray
        :param tableau: a stabilizer tableau
        :type tableau: StabilizerTableau
        :param generator_index: the index of stabilizer generator in the tableau
        :type generator_index: int
        :param emitter_index: the index of chosen emitter register
        :type emitter_index: int
        :param result: the intended type of Pauli
        :type result: str
        :return: nothing
        :rtype: None
        """
        if self.n_emitter == 1:
            return

        if result.lower() == "x":
            assert not np.any(tableau.z_matrix[generator_index])
            emitters = np.nonzero(tableau.x_matrix[generator_index, self.n_photon :])[0]
            control_emitter = emitter_index
            rest_emitters = np.setdiff1d(emitters, [control_emitter])
            for target_emitter in rest_emitters:
                # find two emitters with the desired XX in the generator specified by generator_index
                self._add_one_emitter_cnot(
                    circuit, emitter_depth, control_emitter, target_emitter
                )
                transform.cnot_gate(
                    tableau,
                    self.n_photon + control_emitter,
                    self.n_photon + target_emitter,
                )
        else:
            # only Z
            assert not np.any(tableau.x_matrix[generator_index])
            emitters = np.nonzero(tableau.z_matrix[generator_index, self.n_photon :])[0]
            target_emitter = emitter_index
            rest_emitters = np.setdiff1d(emitters, [target_emitter])
            for control_emitter in rest_emitters:
                # find two emitters with the desired ZZ in the generator specified by generator_index
                self._add_one_emitter_cnot(
                    circuit, emitter_depth, control_emitter, target_emitter
                )
                transform.cnot_gate(
                    tableau,
                    self.n_photon + control_emitter,
                    self.n_photon + target_emitter,
                )

    def _transform_generator_emitters_advanced(
        self, circuit, emitter_depth, tableau, generator_index, result
    ):
        """
        Transform a given generator such that it contains only one type of Pauli (specified by the parameter result)
        on all emitter qubits.
        This function will be used when we can select which emitter for the shortest circuit depth.

        :param circuit: a quantum circuit
        :type circuit: CircuitDAG
        :param emitter_depth: a list of circuit depths for emitter registers
        :type emitter_depth: numpy.ndarray
        :param tableau: a stabilizer tableau
        :type tableau: StabilizerTableau
        :param generator_index: the index of stabilizer generator in the tableau
        :type generator_index: int
        :param result: the intended type of Pauli
        :type result: str
        :return: the index of emitter chosen for the shortest circuit depth
        :rtype: int
        """
        # allow selection of emitters
        if self.n_emitter == 1:
            return 0
        if result.lower() == "x":
            assert not np.any(tableau.z_matrix[generator_index])
            emitters = np.nonzero(tableau.x_matrix[generator_index, self.n_photon :])[0]
            while len(emitters) > 1:
                # find two emitters with the desired Paulis in the generator specified by generator_index
                # if multiple, find these two with the shortest circuit depths
                min_index = np.argmin(emitter_depth[emitters])
                control_emitter = int(emitters[min_index])
                rest_emitters = np.setdiff1d(emitters, [control_emitter])
                min_index = np.argmin(emitter_depth[rest_emitters])
                target_emitter = int(rest_emitters[min_index])
                self._add_one_emitter_cnot(
                    circuit, emitter_depth, control_emitter, target_emitter
                )
                transform.cnot_gate(
                    tableau,
                    self.n_photon + control_emitter,
                    self.n_photon + target_emitter,
                )

                emitters = np.nonzero(
                    tableau.x_matrix[generator_index, self.n_photon :]
                )[0]

        else:
            assert not np.any(tableau.x_matrix[generator_index])
            emitters = np.nonzero(tableau.z_matrix[generator_index, self.n_photon :])[0]
            while len(emitters) > 1:
                # find two emitters with the desired Paulis in the generator specified by generator_index
                # if multiple, find these two with the shortest circuit depths
                min_index = np.argmin(emitter_depth[emitters])
                target_emitter = int(emitters[min_index])
                rest_emitters = np.setdiff1d(emitters, [target_emitter])
                min_index = np.argmin(emitter_depth[rest_emitters])
                control_emitter = int(rest_emitters[min_index])
                self._add_one_emitter_cnot(
                    circuit, emitter_depth, control_emitter, target_emitter
                )
                transform.cnot_gate(
                    tableau,
                    self.n_photon + control_emitter,
                    self.n_photon + target_emitter,
                )

                emitters = np.nonzero(
                    tableau.z_matrix[generator_index, self.n_photon :]
                )[0]
        return int(emitters[0])

    def _add_emitter_photon_cnot(
        self, circuit, emitter_depth, emitter_index, photon_index
    ):
        """
        Add a CNOT gate between two edges

        :param circuit: a quantum circuit
        :type circuit: CircuitDAG
        :return: nothing
        :rtype: None
        """
        emitter_edge = circuit.dag.out_edges(nbunch=f"e{emitter_index}_in", keys=True)
        edge0 = list(emitter_edge)[0]
        photonic_edge = circuit.dag.out_edges(nbunch=f"p{photon_index}_in", keys=True)
        edge1 = list(photonic_edge)[0]

        gate = ops.CNOT(
            control=emitter_index,
            control_type="e",
            target=photon_index,
            target_type="p",
            noise=self._identify_noise(ops.CNOT.__name__, self.noise_model_mapping),
        )
        circuit.insert_at(gate, [edge0, edge1])
        emitter_depth[emitter_index] += 1

    def _find_emitter_indices(self, tableau, generator_index):
        """
        Find nontrivial emitter positions in this generator

        :param tableau: a stabilizer tableau
        :type tableau: StabilizerTableau
        :param generator_index: the index of the generator
        :type generator_index: int
        :return: nontrivial emitter positions in this generator
        :rtype: numpy.ndarray
        """
        generator_sum = (
            tableau.x_matrix[generator_index, self.n_photon :]
            + tableau.z_matrix[generator_index, self.n_photon :]
        )
        return np.nonzero(generator_sum)[0]

    def _single_out_emitter(
        self, circuit, emitter_depth, tableau, generator_index, emitter_index
    ):
        """
        Turn the generator into a single Pauli X. This function is used in the time-reversed measurement.

        :param circuit: a quantum circuit
        :type circuit: CircuitDAG
        :param emitter_depth: a list of circuit depths for emitter registers
        :type emitter_depth: numpy.ndarray
        :param tableau: a stabilizer tableau
        :type tableau: StabilizerTableau
        :param generator_index: the index of stabilizer generator in the tableau
        :type generator_index: int
        :param emitter_index: the index of emitter
        :type emitter_index: int
        :return: nothing
        :rtype: None
        """
        for i in range(self.n_emitter):
            gate_list = self._change_pauli_type(
                tableau, generator_index, self.n_photon + i, "z"
            )
            if len(gate_list) > 0:
                self._add_one_qubit_gate(
                    circuit, emitter_depth, gate_list, self.n_photon + i
                )

        self._transform_generator_emitters(
            circuit, emitter_depth, tableau, generator_index, emitter_index, "z"
        )

        if tableau.phase[generator_index] == 1:
            transform.x_gate(tableau, self.n_photon + emitter_index)
            self._add_one_qubit_gate(
                circuit, emitter_depth, [ops.SigmaX], self.n_photon + emitter_index
            )

    def _add_measurement_cnot_and_reset(self, circuit, emitter_index, photon_index):
        """
        Add a MeasurementCNOTandReset operation from an emitter qubit to a photonic qubit such that no consecutive
        MeasurementCNOTReset is allowed. This operation cannot be added before the photonic qubit is initialized.

        :param circuit: a quantum circuit
        :type circuit: CircuitDAG
        :return: nothing
        :rtype: None
        """
        emitter_edge = circuit.dag.out_edges(nbunch=f"e{emitter_index}_in", keys=True)
        edge0 = list(emitter_edge)[0]
        photonic_edge = circuit.dag.out_edges(nbunch=f"p{photon_index}_in", keys=True)
        edge1 = list(photonic_edge)[0]

        gate = ops.MeasurementCNOTandReset(
            control=circuit.dag.edges[edge0]["reg"],
            control_type="e",
            target=circuit.dag.edges[edge1]["reg"],
            target_type="p",
            noise=self._identify_noise(
                ops.MeasurementCNOTandReset.__name__, self.noise_model_mapping
            ),
        )
        circuit.insert_at(gate, [edge0, edge1])

    def _wrap_noise(self, op, noise_model_mapping):
        """
        A helper function to consolidate noise models for OneQubitWrapper operation

        :param op: a list of operations
        :type op: list[ops.OperationBase]
        :param noise_model_mapping: a dictionary that stores the mapping between an operation
            and its associated noise model
        :type noise_model_mapping: dict
        :return: a list of noise models
        :rtype: list[nm.NoiseBase]
        """
        noise = []
        for each_op in op:
            noise.append(self._identify_noise(each_op.__name__, noise_model_mapping))
        return noise

    def _identify_noise(self, op, noise_model_mapping):
        """
        A helper function to identify the noise model for an operation

        :param op: an operation or its name
        :type op: ops.OperationBase or str
        :param noise_model_mapping: a dictionary that stores the mapping between an operation
            and its associated noise model
        :type noise_model_mapping: dict
        :return: a noise model
        :rtype: nm.NoiseBase
        """
        if type(op) != str:
            op_name = type(op).__name__
        else:
            op_name = op
        if op_name in noise_model_mapping.keys():
            return noise_model_mapping[op_name]
        else:
            return nm.NoNoise()

    @property
    def solver_info(self):
        """
        Return the solver setting

        :return: attributes of the solver
        :rtype: dict
        """

        def op_names(op_list):
            op_name = []
            for op_val in op_list:
                if isinstance(op_val, list):
                    op_name.append([op.__name__ for op in op_val])
                else:
                    op_name.append(op_val.__name__)
            return op_name

        return {
            "solver name": self.name,
            "seed": self.last_seed,
            "One-qubit ops": op_names(self.one_qubit_ops),
        }
