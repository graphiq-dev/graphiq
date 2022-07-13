"""
Evolutionary solver.
"""

import copy
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import warnings
import time

import src.backends.density_matrix.functions as dmf
import src.noise.noise_models as nm

from src.metrics import MetricBase

from src.backends.compiler_base import CompilerBase

from src.solvers import RandomSearchSolver

from src.backends.density_matrix.compiler import DensityMatrixCompiler
from src.circuit import CircuitDAG
from src.metrics import Infidelity

from src.visualizers.density_matrix import density_matrix_bars
from src import ops


class EvolutionarySolver(RandomSearchSolver):
    """
    Implements a rule-based evolutionary search solver.
    This will randomly add/delete/modify operations in the circuit.
    """

    name = "evolutionary-search"
    n_stop = 50  # maximum number of iterations
    n_pop = 50
    n_hof = 5
    tournament_k = 2  # tournament size for selection of the next population

    single_qubit_ops = list(ops.single_qubit_cliffords())

    use_adapt_probability = False

    def __init__(
        self,
        target,
        metric: MetricBase,
        compiler: CompilerBase,
        circuit: CircuitDAG = None,
        n_emitter=1,
        n_photon=1,
        selection_active=False,
        noise_model_mapping={},
        *args,
        **kwargs,
    ):

        super().__init__(target, metric, compiler, circuit, *args, **kwargs)

        self.n_emitter = n_emitter
        self.n_photon = n_photon

        # transformation functions and their relative probabilities
        self.trans_probs = self.initialize_transformation_probabilities()
        self.selection_active = selection_active
        self.noise_model_mapping = noise_model_mapping
        self.p_dist = [0.5] + 11 * [0.1 / 22] + [0.4] + 11 * [0.1 / 22]
        self.e_dist = [0.5] + 11 * [0.02 / 22] + [0.48] + 11 * [0.02 / 22]

    def initialize_transformation_probabilities(self):
        """
        Sets the initial probabilities for selecting the circuit transformations.

        :return: the transformation probabilities for possible transformations
        :rtype: dict
        """
        trans_probs = {
            self.add_emitter_one_qubit_op: 1 / 4,
            self.replace_photon_one_qubit_op: 1 / 4,
            self.remove_op: 1 / 4,
            # self.add_measurement_cnot_and_reset: 1/10,
        }

        if self.n_emitter > 1:
            trans_probs[self.add_emitter_cnot] = 1 / 4

        # normalize the probabilities
        total = np.sum(list(trans_probs.values()))
        for key in trans_probs.keys():
            trans_probs[key] *= 1 / total
        return trans_probs

    def adapt_probabilities(self, iteration: int):
        """
        Changes the probability of selecting circuit transformations at each iteration.
        Generally, transformations that add gates are selected with higher probability at the beginning.
        As the search progresses, transformations that remove gates are selected with higher probability.

        :param iteration: i-th iteration of the search method, which ranges from 0 to n_stop
        :type iteration: int
        :return: nothing
        :rtype: None
        """
        self.trans_probs[self.add_emitter_one_qubit_op] = max(
            self.trans_probs[self.add_emitter_one_qubit_op] - 1 / self.n_stop / 3, 0.01
        )
        self.trans_probs[self.replace_photon_one_qubit_op] = max(
            self.trans_probs[self.replace_photon_one_qubit_op] - 1 / self.n_stop / 3,
            0.01,
        )
        self.trans_probs[self.remove_op] = min(
            self.trans_probs[self.remove_op] + 1 / self.n_stop, 0.99
        )
        if self.n_emitter > 1:
            self.trans_probs[self.add_emitter_cnot] = max(
                self.trans_probs[self.add_emitter_cnot] - 1 / self.n_stop / 3, 0.01
            )

        # normalize the probabilities
        total = np.sum(list(self.trans_probs.values()))
        for key in self.trans_probs.keys():
            self.trans_probs[key] *= 1 / total

    @staticmethod
    def initialization(
        emission_assignment, measurement_assignment, noise_model_mapping={}
    ):
        """
        Initialize a quantum circuit with photon emission, emitter measurements

        :param emission_assignment: which emitter emits which photon
        :type emission_assignment: list[int]
        :param measurement_assignment: which photonic qubit is targeted after measuring each emitter
        :type measurement_assignment: list[int]
        :param noise_model_mapping: a dictionary that stores the mapping between an operation
            and its associated noise model
        :type noise_model_mapping: dict
        :return: nothing
        :rtype: None
        """
        n_photon = len(emission_assignment)
        n_emitter = len(measurement_assignment)
        circuit = CircuitDAG(n_emitter=n_emitter, n_photon=n_photon, n_classical=1)

        for i in range(n_photon):
            # initialize all photon emission gates
            op = ops.CNOT(
                control=emission_assignment[i],
                control_type="e",
                target=i,
                target_type="p",
                noise=EvolutionarySolver._identify_noise(
                    ops.CNOT.__name__, noise_model_mapping
                ),
            )
            op.add_labels("Fixed")
            circuit.add(op)
            # initialize all single-qubit Clifford gate for photonic qubits
            noise = []
            if "Identity" in noise_model_mapping.keys():
                noise.append(noise_model_mapping["Identity"])
            else:
                noise.append(nm.NoNoise())
            if "Hadamard" in noise_model_mapping.keys():
                noise.append(noise_model_mapping["Hadamard"])
            else:
                noise.append(nm.NoNoise())
            op_list = [ops.Identity, ops.Hadamard]
            op = ops.SingleQubitGateWrapper(
                op_list,
                register=i,
                reg_type="p",
                noise=EvolutionarySolver._identify_noise(op_list, noise_model_mapping),
            )
            op.add_labels("Fixed")

            circuit.add(op)

        # initialize all emitter meausurement and reset operations

        for j in range(n_emitter):
            op = ops.MeasurementCNOTandReset(
                control=j,
                control_type="e",
                target=measurement_assignment[j],
                target_type="p",
                noise=EvolutionarySolver._identify_noise(
                    ops.MeasurementCNOTandReset.__name__, noise_model_mapping
                ),
            )
            op.add_labels("Fixed")
            circuit.add(op)
        return circuit

    def solve(self):
        """
        The main function for the solver

        :return: function returns nothing
        :rtype: None
        """

        # TODO: add some logging to see how well it performed at each epoch (and pick n_stop accordingly)

        # Initialize population
        population = []
        if self.circuit is None:
            for j in range(self.n_pop):
                emission_assignment = self.get_emission_assignment(
                    self.n_photon, self.n_emitter
                )
                measurement_assignment = self.get_measurement_assignment(
                    self.n_photon, self.n_emitter
                )

                circuit = self.initialization(
                    emission_assignment,
                    measurement_assignment,
                    self.noise_model_mapping,
                )
                # initialize all population members
                population.append((np.inf, circuit))
        else:
            for j in range(self.n_pop):
                population.append((np.inf, copy.deepcopy(self.circuit)))

        for i in range(self.n_stop):
            for j in range(self.n_pop):
                transformation = np.random.choice(
                    list(self.trans_probs.keys()), p=list(self.trans_probs.values())
                )
                circuit = population[j][1]

                transformation(circuit)

                circuit.validate()

                compiled_state = self.compiler.compile(circuit)
                # this will pass out a density matrix object

                state_data = dmf.partial_trace(
                    compiled_state.data,
                    keep=list(range(self.n_photon)),
                    dims=(self.n_photon + self.n_emitter) * [2],
                )
                score = self.metric.evaluate(state_data, circuit)

                population[j] = (score, circuit)

            self.update_hof(population)
            if EvolutionarySolver.use_adapt_probability:
                self.adapt_probabilities(i)

            if self.selection_active:
                population = self.tournament_selection(population, k=self.tournament_k)

            print(f"Iteration {i} | Best score: {self.hof[0][0]:.4f}")

    @staticmethod
    def _wrap_noise(op, noise_model_mapping):
        """
        A helper function to consolidate noise models for SingleQubitWrapper operation

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
            if each_op.__name__ in noise_model_mapping.keys():
                noise.append(noise_model_mapping[each_op.__name__])
            else:
                noise.append(nm.NoNoise())
        return noise

    @staticmethod
    def _identify_noise(op, noise_model_mapping):
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

    def replace_photon_one_qubit_op(self, circuit):
        """
        Replace one single-qubit Clifford gate applied on a photonic qubit to another one.

        :param circuit: a quantum circuit
        :type circuit: CircuitDAG
        :return: nothing
        :rtype: None
        """
        nodes = circuit.get_node_by_labels(["SingleQubitGateWrapper", "Photonic"])

        if len(nodes) == 0:
            return
        ind = np.random.randint(len(nodes))
        node = list(nodes)[ind]

        old_op = circuit.dag.nodes[node]["op"]

        reg = old_op.register
        ind = np.random.choice(len(self.single_qubit_ops), p=self.p_dist)
        op = self.single_qubit_ops[ind]
        noise = self._wrap_noise(op, self.noise_model_mapping)
        gate = ops.SingleQubitGateWrapper(op, reg_type="p", register=reg, noise=noise)
        gate.add_labels("Fixed")
        # circuit.replace_op(node, gate)
        circuit._open_qasm_update(gate)
        circuit.dag.nodes[node]["op"] = gate

    def add_emitter_one_qubit_op(self, circuit):
        """
        Randomly selects one valid edge on which to add a new single-qubit gate

        :param circuit: a quantum circuit
        :type circuit: CircuitDAG
        :return: nothing
        :rtype: None
        """
        # make sure only selecting emitter qubits and avoiding adding a gate after final measurement or
        # adding two single-qubit Clifford gates in a row
        edges = [
            edge
            for edge in circuit.edge_dict["e"]
            if type(circuit.dag.nodes[edge[1]]["op"]) is not ops.Output
            and type(circuit.dag.nodes[edge[0]]["op"]) is not ops.SingleQubitGateWrapper
            and type(circuit.dag.nodes[edge[1]]["op"]) is not ops.SingleQubitGateWrapper
        ]

        if len(edges) == 0:
            return

        ind = np.random.randint(len(edges))
        edge = list(edges)[ind]

        reg = circuit.dag.edges[edge]["reg"]
        label = edge[2]

        ind = np.random.choice(len(self.single_qubit_ops), p=self.e_dist)
        op = self.single_qubit_ops[ind]
        noise = self._wrap_noise(op, self.noise_model_mapping)
        gate = ops.SingleQubitGateWrapper(op, reg_type="e", register=reg, noise=noise)

        circuit.insert_at(gate, [edge])

    def add_emitter_cnot(self, circuit):
        """
        Randomly selects two valid edges on which to add a new two-qubit gate.
        One edge is selected from all edges, and then the second is selected that maintains proper temporal ordering
        of the operations.

        :param circuit: a quantum circuit
        :type circuit: CircuitDAG
        :return: nothing
        :rtype: None
        """
        possible_edge_pairs = self._select_possible_cnot_position(circuit)
        if len(possible_edge_pairs) == 0:
            warnings.warn("No valid registers to place the two-qubit gate")
            return

        ind = np.random.randint(len(possible_edge_pairs))
        (edge0, edge1) = possible_edge_pairs[ind]

        gate = ops.CNOT(
            control=circuit.dag.edges[edge0]["reg"],
            control_type="e",
            target=circuit.dag.edges[edge1]["reg"],
            target_type="e",
            noise=self._identify_noise(ops.CNOT.__name__, self.noise_model_mapping),
        )

        circuit.insert_at(gate, [edge0, edge1])

    def remove_op(self, circuit, node=None):
        """
        Randomly selects a node in CircuitDAG to remove subject to some restrictions

        :param circuit: a quantum circuit
        :type circuit: CircuitDAG
        :param node: a specified node (by node_id) to be removed
        :type node: int
        :return: nothing
        :rtype: None
        """
        if node is None:
            nodes = circuit.get_node_exclude_labels(["Fixed", "Input", "Output"])

            if len(nodes) == 0:
                return

            ind = np.random.randint(len(nodes))
            node = nodes[ind]
        circuit.remove_op(node)

    def add_measurement_cnot_and_reset(self, circuit):
        """
        Add a MeausurementCNOTandReset operation from an emitter qubit to a photonic qubit such that no consecutive MeasurementCNOTReset is allowed.
        This operation cannot be added before the photonic qubit is initialized.

        :param circuit: a quantum circuit
        :type circuit: CircuitDAG
        :return: nothing
        :rtype: None
        """

        possible_edge_pairs = self._select_possible_measurement_position(circuit)

        if len(possible_edge_pairs) == 0:
            warnings.warn("No valid registers to place the two-qubit gate")
            return

        ind = np.random.randint(len(possible_edge_pairs))
        (edge0, edge1) = possible_edge_pairs[ind]

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

    # helper functions

    @staticmethod
    def _select_possible_cnot_position(circuit):
        """
        Internal helper function to choose all possible pairs of edges to add CNOT gates between two emitter qubits.

        :param circuit: a quantum circuit
        :type circuit: CircuitDAG
        :return: return a list of edge pairs
        :rtype:list[tuple]
        """

        edge_pair = []
        edges = [
            edge
            for edge in circuit.edge_dict["e"]
            if type(circuit.dag.nodes[edge[1]]["op"]) is not ops.Output
        ]

        for edge in edges:
            possible_edges = set(edges) - circuit.find_incompatible_edges(edge)

            for another_edge in possible_edges:
                edge_pair.append((edge, another_edge))

        return edge_pair

    @staticmethod
    def _select_possible_measurement_position(circuit):
        """
        Find all possible positions to add the MeasurementCNOTandReset operation

        :param circuit: a quantum circuit
        :type circuit: CircuitDAG
        :return: a list of edge pairs
        :rtype: list[tuple]
        """
        edge_pair = []
        e_edges = [
            edge
            for edge in circuit.edge_dict["e"]
            if type(circuit.dag.nodes[edge[1]]["op"]) is not ops.Output
            and type(circuit.dag.nodes[edge[1]]["op"])
            is not ops.MeasurementCNOTandReset
            and type(circuit.dag.nodes[edge[0]]["op"]) is not ops.Input
            and type(circuit.dag.nodes[edge[0]]["op"])
            is not ops.MeasurementCNOTandReset
        ]

        p_edges = [
            edge
            for edge in circuit.edge_dict["p"]
            if type(circuit.dag.nodes[edge[1]]["op"]) is not ops.MeasurementCNOTandReset
            and type(circuit.dag.nodes[edge[0]]["op"]) is not ops.Input
        ]

        for edge in e_edges:

            possible_edges = set(p_edges) - circuit.find_incompatible_edges(edge)

            for another_edge in possible_edges:
                edge_pair.append((edge, another_edge))

        return edge_pair

    @staticmethod
    def get_emission_assignment(n_photon, n_emitter):
        """
        Generate a random assignment for the emission source of each photon

        :param n_photon: number of photons
        :type n_photon: int
        :param n_emitter: number of emitters
        :type n_emitter: int
        :return: a list of emitter numbers
        :rtype: list[int]
        """
        if n_emitter == 1:
            return n_photon * [0]
        else:
            assignment = [0]
            available_emitter = [0, 1]
            n_used_emitter = 1
            for i in range(1, n_photon):
                if n_photon - i == n_emitter - n_used_emitter:
                    assignment.append(n_used_emitter)
                    n_used_emitter = n_used_emitter + 1
                    if n_used_emitter < n_emitter:
                        available_emitter.append(n_used_emitter)
                else:
                    ind = np.random.randint(len(available_emitter))
                    assignment.append(ind)
                    if ind == n_used_emitter and n_used_emitter < n_emitter:
                        n_used_emitter = n_used_emitter + 1
                        if n_used_emitter < n_emitter:
                            available_emitter.append(ind + 1)

            return assignment

    @staticmethod
    def get_measurement_assignment(n_photon, n_emitter):
        """
        Generate a random assignment for the target of measurement-based control X gate after measuring each emitter qubit

        :param n_photon: number of photons
        :type n_photon: int
        :param n_emitter: number of emitters
        :type n_emitter: int
        :return: a list of photon numbers
        :rtype: list[int]
        """
        return np.random.randint(n_photon, size=n_emitter).tolist()

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

        def transition_names(transition_dict):
            transition_name = {}
            for key, value in transition_dict.items():
                transition_name[key.__name__] = value
            return transition_name

        return {
            "solver name": self.name,
            "Max iteration #": self.n_stop,
            "Population size": self.n_pop,
            "seed": self.last_seed,
            "Single qubit ops": op_names(self.single_qubit_ops),
            "Transition probabilities": transition_names(self.trans_probs),
        }
