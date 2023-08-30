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

from src.metrics import MetricBase
from src.solvers import SolverBase
from src.circuit.circuit_dag import CircuitDAG
from src.circuit import ops
from src.io import IO


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
        if target.rep_type is not "s":
            target.convert_representation("s")
        tableau = target.rep_data.data.to_stabilizer()
        self.n_emitter = self.determine_n_emitters(tableau)
        self.n_photon = tableau.n_qubits
        self.noise_simulation = True

        if noise_model_mapping is None:
            noise_model_mapping = {"e": dict(), "p": dict(), "ee": dict(), "ep": dict()}
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

        stabilizer_tableau = self.target.rep_data.data.to_stabilizer()

        # add emitter qubits in |0> state
        for _ in range(self.n_emitter):
            stabilizer_tableau = sfs.insert_qubit(
                stabilizer_tableau, stabilizer_tableau.n_qubits
            )

        # main loop
        for j in range(self.n_photon, 0, -1):
            # transform the stabilizers into echelon gauge
            stabilizer_tableau = sfs.rref(stabilizer_tableau)
            height_list = height.height_func_list(
                stabilizer_tableau.x_matrix, stabilizer_tableau.z_matrix
            )

            height_list = [0] + height_list
            if height_list[j] < height_list[j - 1]:
                # apply time-reversed measurement and update the tableau
                self._time_reversed_measurement(circuit, stabilizer_tableau, j - 1)
                stabilizer_tableau = sfs.rref(stabilizer_tableau)

            # apply photon-absorption and update the stabilizer tableau
            self._add_photon_absorption(circuit, stabilizer_tableau, j - 1)

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

        self._add_gates_from_str(circuit, stabilizer_tableau, inverse_circuit)

        # correct the phase of generators that act only on emitter qubits
        for i in range(self.n_emitter):
            if stabilizer_tableau.phase[self.n_photon + i] == 1:
                transform.x_gate(stabilizer_tableau, self.n_photon + i)
                self._add_one_qubit_gate(circuit, [ops.SigmaX], self.n_photon + i)

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

    def _time_reversed_measurement(self, circuit, tableau, photon_index):
        """
        Apply the time-reversed measurement

        :param circuit: a quantum circuit
        :type circuit: CircuitDAG
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

        self._single_out_emitter(circuit, tableau, generator_index, emitter_index)

        transform.hadamard_gate(tableau, self.n_photon + emitter_index)
        self._add_measurement_cnot_and_reset(circuit, emitter_index, photon_index)

        # transform this generator so that it acts nontrivially on only one emitter and that Pauli is X
        transform.cnot_gate(tableau, self.n_photon + emitter_index, photon_index)

    def _add_photon_absorption(self, circuit, tableau, photon_index):
        """
        Add a photon absorption (time-reversed photon emission) event

        :param circuit: a quantum circuit
        :type circuit: CircuitDAG
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
        self._add_one_qubit_gate(circuit, gate_list, photon_index)

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
            self._add_one_qubit_gate(circuit, gate_list, self.n_photon + i)

        # now this generator contains only ZZ on emitter qubits
        # transform this generator so that for emitter qubits, there is only one Z.
        self._transform_generator_emitters(
            circuit, tableau, generator_index, emitter_index
        )

        # if the phase of this generator is -1, flip it by adding X gate to the emitter qubit
        if tableau.phase[generator_index] == 1:
            transform.x_gate(tableau, self.n_photon + emitter_index)
            self._add_one_qubit_gate(
                circuit, [ops.SigmaX], self.n_photon + emitter_index
            )

        # emit (absorb) this photon by the chosen emitter
        self._add_emitter_photon_cnot(circuit, emitter_index, photon_index)
        transform.cnot_gate(tableau, self.n_photon + emitter_index, photon_index)

        # transform all generators (except the chosen generator) with Z on this photonic register
        # by eliminating Z
        pauli_z_list = sfs.one_pauli_type_finder(
            tableau.x_matrix, tableau.z_matrix, [0, photon_index], "z"
        )

        pauli_z_list = np.setdiff1d(pauli_z_list, [generator_index])
        for i in pauli_z_list:
            # use left multiplication instead of right multiplication
            sfs.tab_row_sum(tableau, generator_index, i)

    def _add_gates_from_str(self, circuit, tableau, gate_str_list):
        """
        Add gates to disentangle all emitter qubits. This is used in the last step.

        :param circuit: a quantum circuit
        :type circuit: CircuitDAG
        :param tableau: a stabilizer tableau
        :type tableau: StabilizerTableau
        :param gate_str_list: a list of gates to be applied
        :type gate_str_list: list[(str, int) or (str, int, int)]
        :return: nothing
        :rtype: None
        """

        for gate in gate_str_list:
            if gate[0] == "H":
                self._add_one_qubit_gate(circuit, [ops.Hadamard], gate[1])
                transform.hadamard_gate(tableau, gate[1])
            elif gate[0] == "P":
                # add the inverse of the phase gate
                self._add_one_qubit_gate(circuit, [ops.SigmaZ, ops.Phase], gate[1])
                transform.phase_gate(tableau, gate[1])
            elif gate[0] == "X":
                self._add_one_qubit_gate(circuit, [ops.SigmaX], gate[1])
                transform.x_gate(tableau, gate[1])
            elif gate[0] == "CNOT":
                self._add_one_emitter_cnot(
                    circuit, gate[1] - self.n_photon, gate[2] - self.n_photon
                )
                transform.cnot_gate(tableau, gate[1], gate[2])
            elif gate[0] == "CZ":
                self._add_one_qubit_gate(circuit, [ops.Hadamard], gate[2])
                transform.hadamard_gate(tableau, gate[2])
                self._add_one_emitter_cnot(
                    circuit, gate[1] - self.n_photon, gate[2] - self.n_photon
                )
                transform.cnot_gate(tableau, gate[1], gate[2])
                self._add_one_qubit_gate(circuit, [ops.Hadamard], gate[2])
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

    def _add_one_qubit_gate(self, circuit, gate_list, index):
        """
        Add a one-qubit gate to the circuit

        :param circuit: a quantum circuit
        :type circuit: CircuitDAG
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
                return
        else:
            # simplify the gate to be one of the 24 local Clifford gates
            gate_list = ops.simplify_local_clifford(gate_list)
            if gate_list == [ops.Identity, ops.Identity]:
                return
        if reg_type == "e":
            noise = self._wrap_noise(gate_list, self.noise_model_mapping["e"])
        else:
            noise = self._wrap_noise(gate_list, self.noise_model_mapping["p"])
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

    def _add_one_emitter_cnot(self, circuit, control_emitter, target_emitter):
        """
        Add a CNOT between two emitter qubits

        :param circuit: a quantum circuit
        :type circuit: CircuitDAG
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
            noise=self._identify_noise(ops.CNOT, self.noise_model_mapping["ee"]),
        )
        circuit.insert_at(gate, [edge0, edge1])

    def _transform_generator_emitters(
        self, circuit, tableau, generator_index, emitter_index
    ):
        """
        Transform a given generator such that it contains only one Pauli Z on emitter qubits

        :param circuit: a quantum circuit
        :type circuit: CircuitDAG
        :param tableau: a stabilizer tableau
        :type tableau: StabilizerTableau
        :param generator_index: the index of stabilizer generator in the tableau
        :type generator_index: int
        :param emitter_index: the index of chosen emitter register
        :type emitter_index: int
        :return: nothing
        :rtype: None
        """

        if self.n_emitter == 1:
            return

        # only Z
        assert not np.any(tableau.x_matrix[generator_index])
        emitters = np.nonzero(tableau.z_matrix[generator_index, self.n_photon :])[0]
        target_emitter = emitter_index
        rest_emitters = np.setdiff1d(emitters, [target_emitter])
        for control_emitter in rest_emitters:
            # find two emitters with ZZ in the generator specified by generator_index
            # add CNOT to remove one Z
            self._add_one_emitter_cnot(circuit, control_emitter, target_emitter)
            transform.cnot_gate(
                tableau,
                self.n_photon + control_emitter,
                self.n_photon + target_emitter,
            )

    def _transform_generator_emitters_advanced(self, circuit, tableau, generator_index):
        """
        Transform a given generator such that it contains only one Pauli Z on emitter qubits
        This function will be used when we can select which emitter for the shortest circuit depth.

        :param circuit: a quantum circuit
        :type circuit: CircuitDAG
        :param tableau: a stabilizer tableau
        :type tableau: StabilizerTableau
        :param generator_index: the index of stabilizer generator in the tableau
        :type generator_index: int
        :return: the index of emitter chosen for the shortest circuit depth
        :rtype: int
        """

        # wait for the circuit class refactoring
        emitter_depth = circuit.register_depth["e"]
        # allow selection of emitters
        if self.n_emitter == 1:
            return 0

        assert not np.any(tableau.x_matrix[generator_index])
        emitters = np.nonzero(tableau.z_matrix[generator_index, self.n_photon :])[0]
        while len(emitters) > 1:
            # find two emitters with ZZ in the generator specified by generator_index
            # if multiple, find these two with the shortest circuit depths
            min_index = np.argmin(emitter_depth[emitters])
            target_emitter = int(emitters[min_index])
            rest_emitters = np.setdiff1d(emitters, [target_emitter])
            min_index = np.argmin(emitter_depth[rest_emitters])
            control_emitter = int(rest_emitters[min_index])
            self._add_one_emitter_cnot(circuit, control_emitter, target_emitter)
            transform.cnot_gate(
                tableau,
                self.n_photon + control_emitter,
                self.n_photon + target_emitter,
            )

            emitters = np.nonzero(tableau.z_matrix[generator_index, self.n_photon :])[0]
        return int(emitters[0])

    def _add_emitter_photon_cnot(self, circuit, emitter_index, photon_index):
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
            noise=self._identify_noise(ops.CNOT, self.noise_model_mapping["ep"]),
        )
        gate.add_labels("Fixed")
        circuit.insert_at(gate, [edge0, edge1])

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

    def _single_out_emitter(self, circuit, tableau, generator_index, emitter_index):
        """
        Turn the chosen generator into a single Pauli Z. This function is used in the time-reversed measurement.

        :param circuit: a quantum circuit
        :type circuit: CircuitDAG
        :param tableau: a stabilizer tableau
        :type tableau: StabilizerTableau
        :param generator_index: the index of stabilizer generator in the tableau
        :type generator_index: int
        :param emitter_index: the index of emitter
        :type emitter_index: int
        :return: nothing
        :rtype: None
        """

        # first transform this generator so that it only contains Zs.
        for i in range(self.n_emitter):
            gate_list = self._change_pauli_type(
                tableau, generator_index, self.n_photon + i, "z"
            )
            if len(gate_list) > 0:
                self._add_one_qubit_gate(circuit, gate_list, self.n_photon + i)

        # disentangle the chosen emitter from the rest
        self._transform_generator_emitters(
            circuit, tableau, generator_index, emitter_index
        )

        if tableau.phase[generator_index] == 1:
            transform.x_gate(tableau, self.n_photon + emitter_index)
            self._add_one_qubit_gate(
                circuit, [ops.SigmaX], self.n_photon + emitter_index
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
                ops.MeasurementCNOTandReset, self.noise_model_mapping["ep"]
            ),
        )
        gate.add_labels("Fixed")
        circuit.insert_at(gate, [edge0, edge1])

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
