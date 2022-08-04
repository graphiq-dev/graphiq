"""
Evolutionary solver which includes a random search solver as a special case.
This solver is based on certain physical rules imposed by a platform.
One can define these rules via the allowed DAG transformations.
"""
import copy
import numpy as np
import matplotlib.pyplot as plt
import warnings

import src.backends.density_matrix.functions as dmf
import src.noise.noise_models as nm
import pandas as pd
from src.metrics import MetricBase

from src.backends.compiler_base import CompilerBase
from src.backends.density_matrix.compiler import DensityMatrixCompiler
from src.solvers import RandomSearchSolver
from src.circuit import CircuitDAG
from src.metrics import MetricBase
from src.metrics import Infidelity

from src.visualizers.density_matrix import density_matrix_bars
from src.io import IO
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

    one_qubit_ops = list(ops.one_qubit_cliffords())

    use_adapt_probability = False

    def __init__(
        self,
        target,
        metric: MetricBase,
        compiler: CompilerBase,
        circuit: CircuitDAG = None,
        io: IO = None,
        n_emitter=1,
        n_photon=1,
        selection_active=False,
        save_openqasm: str = "none",
        noise_model_mapping=None,
        *args,
        **kwargs,
    ):
        """

        :param target:
        :param metric:
        :param compiler:
        :param circuit:
        :param io:
        :param n_emitter:
        :param n_photon:
        :param selection_active:
        :param save_openqasm:
        :param args:
        :param kwargs:
        """
        super().__init__(target, metric, compiler, circuit, io, *args, **kwargs)

        self.n_emitter = n_emitter
        self.n_photon = n_photon

        # transformation functions and their relative probabilities
        self.trans_probs = self.initialize_transformation_probabilities()
        self.selection_active = selection_active
        self.noise_simulation = True
        if noise_model_mapping is None or type(noise_model_mapping) is not dict:
            noise_model_mapping = {}
            self.noise_simulation = False
        self.noise_model_mapping = noise_model_mapping

        self.save_openqasm = save_openqasm

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

        return self._normalize_trans_prob(trans_probs)

    @staticmethod
    def _normalize_trans_prob(trans_probs):
        """
        Helper function to normalize the transformation probabilities

        :param trans_probs: transformation probabilities
        :type trans_probs: dict
        :return: transformation probabilities after normalization
        """
        total = np.sum(list(trans_probs.values()))
        for key in trans_probs.keys():
            trans_probs[key] *= 1 / total
        return trans_probs

    def set_allowed_transformations(self, allowed_transformations):
        """
        Set allowed transformation and corresponding probabilities

        :param allowed_transformations: a dictionary of all allowed transformation and its probabilities
        :type allowed_transformations: dict
        :return: nothing
        :rtype: None
        """

        self.trans_probs = self._normalize_trans_prob(allowed_transformations)

    def update_emitter_one_qubit_gate_probs(self, e_prob_list):
        """
        Update the probability distribution of one-qubit Clifford gates for emitter qubits

        :param e_prob_list: a list of probabilities
        :type e_prob_list: list[float]
        :return: nothing
        :rtype: None
        """
        assert len(e_prob_list) == len(self.one_qubit_ops)
        e_prob_list = np.array(e_prob_list)
        assert (e_prob_list >= 0).all()
        total_prob = sum(e_prob_list)
        assert total_prob > 0
        self.e_dist = list(e_prob_list / total_prob)

    def update_photonic_one_qubit_gate_probs(self, p_prob_list):
        """
        Update the probability distribution of one-qubit Clifford gates for photonic qubits

        :param p_prob_list: a list of probabilities
        :type p_prob_list: list[float]
        :return: nothing
        :rtype: None
        """
        assert len(p_prob_list) == len(self.one_qubit_ops)
        p_prob_list = np.array(p_prob_list)
        assert (p_prob_list >= 0).all()
        total_prob = sum(p_prob_list)
        assert total_prob > 0
        self.p_dist = list(p_prob_list / total_prob)

    def adapt_probabilities(self, iteration: int):
        """
        Changes the probability of selecting circuit transformations at each iteration.
        Generally, transformations that add gates are selected with higher probability at the beginning.
        As the search progresses, transformations that remove gates are selected with higher probability.

        TODO: check whether the input iteration is needed.

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
            # initialize all one-qubit Clifford gate for photonic qubits
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
            op = ops.OneQubitGateWrapper(
                op_list,
                register=i,
                reg_type="p",
                noise=EvolutionarySolver._wrap_noise(op_list, noise_model_mapping),
            )
            op.add_labels("Fixed")

            circuit.add(op)

        # initialize all emitter measurement and reset operations

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

    """ Main solver algorithm """

    def solve(self):
        """
        The main function for the solver

        :return: function returns nothing
        :rtype: None
        """

        self.compiler.noise_simulation = self.noise_simulation
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

            # self.adapt_probabilities(i)

            # this should be the last thing performed *prior* to selecting a new population (after updating HoF)
            self.update_logs(population=population, iteration=i)
            self.save_circuits(population=population, hof=self.hof, iteration=i)

            if self.selection_active:
                population = self.tournament_selection(population, k=self.tournament_k)

            print(f"Iteration {i} | Best score: {self.hof[0][0]:.4f}")

        self.logs_to_df()  # convert the logs to a DataFrame

    """ Logging and saving openQASM strings """

    def update_logs(self, population: list, iteration: int):
        """
        Updates the log table, which tracks cost function values through solver iterations.

        :param population: population list for the i-th iteration, as a list of tuples (score, circuit)
        :param iteration: iteration integer, from 0 to n_stop-1
        :return:
        """
        # get the scores from the population/hof as a list
        scores_pop = list(zip(*population))[0]
        scores_hof = list(zip(*self.hof))[0]

        depth_pop = [circuit.depth for (_, circuit) in population]
        depth_hof = [circuit.depth for (_, circuit) in self.hof]

        self.logs["population"].append(
            dict(
                iteration=iteration,
                cost_mean=np.mean(scores_pop),
                cost_variance=np.var(scores_pop),
                cost_min=np.min(scores_pop),
                cost_max=np.max(scores_pop),
                depth_mean=np.mean(depth_pop),
                depth_variance=np.var(depth_pop),
                depth_min=np.min(depth_pop),
                depth_max=np.max(depth_pop),
            )
        )

        self.logs["hof"].append(
            dict(
                iteration=iteration,
                cost_mean=np.mean(scores_hof),
                cost_variance=np.var(scores_hof),
                cost_min=np.min(scores_hof),
                cost_max=np.max(scores_hof),
                depth_mean=np.mean(depth_hof),
                depth_variance=np.var(depth_hof),
                depth_min=np.min(depth_hof),
                depth_max=np.max(depth_hof),
            )
        )

    def save_circuits(self, population: list, hof: list, iteration: int = -1):
        """
        Saves the population and/or the HoF circuits over iterations as openQASM strings.

        :param population: list of (score, circuit) pairs
        :param hof: list of (score, circuit) pairs
        :param iteration: integer step in the solver algorithm
        :return:
        """
        if self.io is None or self.save_openqasm == "none":
            return

        elif self.io is not None and self.save_openqasm == "hof":
            self._save_hof(hof, iteration=iteration)

        elif self.io is not None and self.save_openqasm == "pop":
            self._save_pop(population, iteration=iteration)

        elif self.io is not None and self.save_openqasm == "both":
            self._save_pop(population, iteration=iteration)
            self._save_hof(hof, iteration=iteration)
        return

    def _save_pop(self, pop, iteration):
        for i, (score, circuit) in enumerate(pop):
            name = f"pop/iteration{iteration}_pop{i}.txt"
            self.io.save_json(circuit.to_openqasm(), name)

    def _save_hof(self, hof, iteration):
        for i, (score, circuit) in enumerate(hof):
            name = f"hof/iteration{iteration}_hof{i}.txt"
            self.io.save_json(circuit.to_openqasm(), name)

    def logs_to_df(self):
        """Converts each logs (population, hof, etc.) to a pandas DataFrame for easier visualization/saving"""
        for key, val in self.logs.items():
            self.logs[key] = pd.DataFrame(val)

    @staticmethod
    def _wrap_noise(op, noise_model_mapping):
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
            noise.append(
                EvolutionarySolver._identify_noise(
                    each_op.__name__, noise_model_mapping
                )
            )
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

    """ Circuit transformations """

    def replace_photon_one_qubit_op(self, circuit):
        """
        Replace one one-qubit Clifford gate applied on a photonic qubit to another one.

        :param circuit: a quantum circuit
        :type circuit: CircuitDAG
        :return: nothing
        :rtype: None
        """
        nodes = circuit.get_node_by_labels(["OneQubitGateWrapper", "Photonic"])

        if len(nodes) == 0:
            return
        ind = np.random.randint(len(nodes))
        node = list(nodes)[ind]

        old_op = circuit.dag.nodes[node]["op"]

        reg = old_op.register
        ind = np.random.choice(len(self.one_qubit_ops), p=self.p_dist)
        op = self.one_qubit_ops[ind]
        noise = self._wrap_noise(op, self.noise_model_mapping)
        gate = ops.OneQubitGateWrapper(op, reg_type="p", register=reg, noise=noise)
        gate.add_labels("Fixed")
        # circuit.replace_op(node, gate)
        circuit._openqasm_update(gate)
        circuit.dag.nodes[node]["op"] = gate

    def add_emitter_one_qubit_op(self, circuit):
        """
        Randomly selects one valid edge on which to add a new one-qubit gate

        :param circuit: a quantum circuit
        :type circuit: CircuitDAG
        :return: nothing
        :rtype: None
        """
        # make sure only selecting emitter qubits and avoiding adding a gate after final measurement or
        # adding two one-qubit Clifford gates in a row
        edges = [
            edge
            for edge in circuit.edge_dict["e"]
            if type(circuit.dag.nodes[edge[1]]["op"]) is not ops.Output
            and type(circuit.dag.nodes[edge[0]]["op"]) is not ops.OneQubitGateWrapper
            and type(circuit.dag.nodes[edge[1]]["op"]) is not ops.OneQubitGateWrapper
        ]

        if len(edges) == 0:
            return

        ind = np.random.randint(len(edges))
        edge = list(edges)[ind]

        reg = circuit.dag.edges[edge]["reg"]
        label = edge[2]

        ind = np.random.choice(len(self.one_qubit_ops), p=self.e_dist)
        op = self.one_qubit_ops[ind]
        noise = self._wrap_noise(op, self.noise_model_mapping)
        gate = ops.OneQubitGateWrapper(op, reg_type="e", register=reg, noise=noise)

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
        Add a MeausurementCNOTandReset operation from an emitter qubit to a photonic qubit such that no consecutive
        MeasurementCNOTReset is allowed. This operation cannot be added before the photonic qubit is initialized.

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
            "One-qubit ops": op_names(self.one_qubit_ops),
            "Transition probabilities": transition_names(self.trans_probs),
        }
