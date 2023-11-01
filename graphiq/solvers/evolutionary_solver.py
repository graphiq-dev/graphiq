"""
Evolutionary solver which includes a random search solver as a special case.
This solver is based on certain physical rules imposed by a platform.
One can define these rules via the allowed DAG transformations.

"""

import warnings

import numpy as np
import pandas as pd

import graphiq.noise.noise_models as nm
from graphiq.backends.compiler_base import CompilerBase
from graphiq.circuit import ops
from graphiq.circuit.circuit_dag import CircuitDAG
from graphiq.io import IO
from graphiq.metrics import MetricBase
from graphiq.solvers.solver_base import RandomSearchSolver, RandomSearchSolverSetting


class EvolutionarySolverSetting(RandomSearchSolverSetting):
    """
    A class to store the solver setting of an EvolutionarySearchSolver

    """

    def __init__(
            self,
            n_hof=5,
            n_stop=50,
            n_pop=50,
            tournament_k=2,
            selection_active=False,
            use_adapt_probability=False,
            verbose=False,
            save_openqasm: str = "none",
    ):
        super().__init__(n_hof=n_hof, n_stop=n_stop, n_pop=n_pop)
        self._tournament_k = tournament_k
        self._selection_active = selection_active
        self._use_adapt_probability = use_adapt_probability
        self._save_openqasm = save_openqasm
        self._verbose = verbose

    @property
    def tournament_k(self):
        return self._tournament_k

    @tournament_k.setter
    def tournament_k(self, value):
        assert type(value) == int
        self._tournament_k = value

    @property
    def selection_active(self):
        return self._selection_active

    @selection_active.setter
    def selection_active(self, value):
        assert type(value) == bool
        self._selection_active = value

    @property
    def use_adapt_probability(self):
        return self._use_adapt_probability

    @use_adapt_probability.setter
    def use_adapt_probability(self, value):
        assert type(value) == bool
        self._use_adapt_probability = value

    @property
    def verbose(self):
        return self._verbose

    @verbose.setter
    def verbose(self, value):
        assert type(value) == bool
        self._verbose = value

    @property
    def save_openqasm(self):
        return self._save_openqasm

    @save_openqasm.setter
    def save_openqasm(self, value):
        assert type(value) == str
        self._save_openqasm = value

    def __str__(self):
        s = super().__str__()
        s += f"selection_active = {self.selection_active} \n"
        s += f"use_adapt_probability = {self.use_adapt_probability} \n"
        s += f"tournament_k = {self.tournament_k} \n"
        s += f"save_openqasm = {self.save_openqasm}\n"
        return s


class EvolutionarySolver(RandomSearchSolver):
    """
    Implements a rule-based evolutionary search solver.
    This will randomly add/delete/modify operations in the circuit.
    """

    name = "evolutionary-search"

    one_qubit_ops = list(ops.one_qubit_cliffords())

    def __init__(
            self,
            target,
            metric: MetricBase,
            compiler: CompilerBase,
            circuit: CircuitDAG = None,
            io: IO = None,
            n_emitter=1,
            n_photon=1,
            solver_setting=EvolutionarySolverSetting(),
            noise_model_mapping=None,
            *args,
            **kwargs,
    ):
        """
        Initialize an EvolutionarySolver object. When selection_active and use_adapt_probability are False, this solver
        reduces to a random search solver.

        :param target: target quantum state
        :type target: QuantumState
        :param metric: metric (cost) function to minimize
        :type metric: MetricBase
        :param compiler: compiler backend to use when simulating quantum circuits
        :type compiler: CompilerBase
        :param circuit: (optional) initial circuit
        :type circuit: CircuitDAG
        :param io: input/output object for saving logs, intermediate results, circuits, etc.
        :type io: IO
        :param n_emitter: number of emitter registers to maintain in the circuit
        :type n_emitter: int
        :param n_photon: number of photon registers to maintain in the circuit
        :type n_photon: int
        :param noise_model_mapping: a dictionary that associates each operation type to a noise model
        :type noise_model_mapping: dict
        """
        super().__init__(
            target=target,
            metric=metric,
            compiler=compiler,
            circuit=circuit,
            io=io,
            solver_setting=solver_setting,
            *args,
            **kwargs,
        )

        self.n_emitter = n_emitter
        self.n_photon = n_photon

        # transformation functions and their relative probabilities
        self.trans_probs = self.initialize_transformation_probabilities()

        if noise_model_mapping is None:
            noise_model_mapping = {"e": dict(), "p": dict(), "ee": dict(), "ep": dict()}
            self.noise_simulation = False
        elif type(noise_model_mapping) is not dict:
            raise TypeError(
                f"Datatype {type(noise_model_mapping)} is not a valid noise_model_mapping. "
                f"noise_model_mapping should be a dict or None"
            )
        else:
            self.noise_simulation = True

        self.noise_model_mapping = noise_model_mapping
        self.setting = solver_setting

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

    def adapt_probabilities(self):
        """
        Changes the probability of selecting circuit transformations at each iteration.
        Generally, transformations that add gates are selected with higher probability at the beginning.
        As the search progresses, transformations that remove gates are selected with higher probability.

        :return: nothing
        :rtype: None
        """
        # TODO: check whether the input iteration is needed.
        self.trans_probs[self.add_emitter_one_qubit_op] = max(
            self.trans_probs[self.add_emitter_one_qubit_op]
            - 1 / self.setting.n_stop / 3,
            0.01,
        )
        self.trans_probs[self.replace_photon_one_qubit_op] = max(
            self.trans_probs[self.replace_photon_one_qubit_op]
            - 1 / self.setting.n_stop / 3,
            0.01,
        )
        self.trans_probs[self.remove_op] = min(
            self.trans_probs[self.remove_op] + 1 / self.setting.n_stop, 0.99
        )
        if self.n_emitter > 1:
            self.trans_probs[self.add_emitter_cnot] = max(
                self.trans_probs[self.add_emitter_cnot] - 1 / self.setting.n_stop / 3,
                0.01,
            )

        # normalize the probabilities
        total = np.sum(list(self.trans_probs.values()))
        for key in self.trans_probs.keys():
            self.trans_probs[key] *= 1 / total

    def initialization(
            self, emission_assignment, measurement_assignment, noise_model_mapping=None
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
        :return: a circuit specified by emission and measurement assignments
        :rtype: CircuitDAG
        """
        if noise_model_mapping is None:
            noise_model_mapping = {"e": dict(), "p": dict(), "ee": dict(), "ep": dict()}

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
                noise=self._identify_noise(ops.CNOT, noise_model_mapping["ep"]),
            )
            op.add_labels("Fixed")
            circuit.add(op)
            # initialize all one-qubit Clifford gate for photonic qubits
            noise = []
            if "Identity" in noise_model_mapping["p"].keys():
                noise.append(noise_model_mapping["p"]["Identity"])
            else:
                noise.append(nm.NoNoise())
            if "Hadamard" in noise_model_mapping["p"].keys():
                noise.append(noise_model_mapping["p"]["Hadamard"])
            else:
                noise.append(nm.NoNoise())
            op_list = [ops.Identity, ops.Hadamard]
            op = ops.OneQubitGateWrapper(
                op_list,
                register=i,
                reg_type="p",
                noise=self._wrap_noise(op_list, noise_model_mapping["p"]),
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
                noise=self._identify_noise(
                    ops.MeasurementCNOTandReset, noise_model_mapping["ep"]
                ),
            )
            op.add_labels("Fixed")
            circuit.add(op)
        return circuit

    def population_initialization(self):
        """
        Initialize population

        :return: an initial population
        :rtype: list
        """
        population = []
        if self.circuit is None:
            for j in range(self.setting.n_pop):
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
            for j in range(self.setting.n_pop):
                population.append((np.inf, self.circuit.copy()))

        return population

    """ Main solver algorithm """

    def solve(self):
        """
        The main function for the solver

        :return: function returns nothing
        :rtype: None
        """

        self.compiler.noise_simulation = self.noise_simulation

        population = self.population_initialization()

        for i in range(self.setting.n_stop):
            for j in range(self.setting.n_pop):
                # choose a random transformation from allowed transformations
                transformation = np.random.choice(
                    list(self.trans_probs.keys()), p=list(self.trans_probs.values())
                )
                # evolve the chosen circuit
                circuit = population[j][1]
                transformation(circuit)
                circuit.validate()

                # compile the output state of the circuit
                compiled_state = self.compiler.compile(circuit)
                # trace out emitter qubits
                compiled_state.partial_trace(
                    keep=list(range(self.n_photon)),
                    dims=(self.n_photon + self.n_emitter) * [2],
                )
                # evaluate the metric
                score = self.metric.evaluate(compiled_state, circuit)

                population[j] = (score, circuit)

            self.update_hof(population)
            if self.setting.use_adapt_probability:
                self.adapt_probabilities()

            # this should be the last thing performed *prior* to selecting a new population (after updating HoF)
            self.update_logs(population=population, iteration=i)
            self.save_circuits(population=population, hof=self.hof, iteration=i)

            if self.setting.selection_active:
                population = self.tournament_selection(
                    population, k=self.setting.tournament_k
                )
            if self.setting.verbose:
                print(f"Iteration {i} | Best score: {self.hof[0][0]:.6f}")

        self.logs_to_df()  # convert the logs to a DataFrame
        self.result = (self.hof[0][0], self.hof[0][1])

    """ Logging and saving openQASM strings """

    def update_logs(self, population: list, iteration: int):
        """
        Updates the log table, which tracks cost function values through solver iterations.

        :param population: population list for the i-th iteration, as a list of tuples (score, circuit)
        :type population: list
        :param iteration: iteration integer, from 0 to n_stop-1
        :type iteration: int
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
        if self.io is None or self.setting.save_openqasm == "none":
            return

        elif self.setting.save_openqasm == "hof":
            self._save_hof(hof, iteration=iteration)

        elif self.setting.save_openqasm == "pop":
            self._save_pop(population, iteration=iteration)

        elif self.setting.save_openqasm == "both":
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

    """ Circuit transformations """

    def replace_photon_one_qubit_op(self, circuit):
        """
        Replace a one-qubit Clifford gate that is applied on a photonic qubit by another one.

        :param circuit: a quantum circuit
        :type circuit: CircuitDAG
        :return: nothing
        :rtype: None
        """
        nodes = circuit.get_node_by_labels(["OneQubitGateWrapper", "Photonic"])

        if len(nodes) == 0:
            return
        # find a random one-qubit operation on a photonic qubit
        ind = np.random.randint(len(nodes))
        node = list(nodes)[ind]
        old_op = circuit.dag.nodes[node]["op"]
        reg = old_op.register

        # select a random local Clifford gate
        ind = np.random.choice(len(self.one_qubit_ops), p=self.p_dist)
        op = self.one_qubit_ops[ind]
        noise = self._wrap_noise(op, self.noise_model_mapping["p"])
        gate = ops.OneQubitGateWrapper(op, reg_type="p", register=reg, noise=noise)
        gate.add_labels("Fixed")
        circuit.replace_op(node, gate)

    def replace_emitter_one_qubit_op(self, circuit):
        """
        Replace a one-qubit Clifford gate that is applied on an emitter qubit by another one.

        :param circuit: a quantum circuit
        :type circuit: CircuitDAG
        :return: nothing
        :rtype: None
        """
        nodes = circuit.get_node_by_labels(["OneQubitGateWrapper", "Emitter"])

        if len(nodes) == 0:
            return
        # find a random one-qubit operation on an emitter qubit
        ind = np.random.randint(len(nodes))
        node = list(nodes)[ind]
        old_op = circuit.dag.nodes[node]["op"]
        reg = old_op.register

        # select a random local Clifford gate
        ind = np.random.choice(len(self.one_qubit_ops), p=self.e_dist)
        op = self.one_qubit_ops[ind]
        noise = self._wrap_noise(op, self.noise_model_mapping["e"])
        gate = ops.OneQubitGateWrapper(op, reg_type="e", register=reg, noise=noise)
        circuit.replace_op(node, gate)

    def add_photon_one_qubit_op(self, circuit):
        """
        Add a single-qubit Clifford gate to a photonic qubit

        :param circuit: a quantum circuit
        :type circuit: CircuitDAG
        :return: function returns nothing
        :rtype: None
        """

        # make sure only selecting a photonic qubit after emission
        edges = [
            edge
            for edge in circuit.edge_dict["p"]
            if type(circuit.dag.nodes[edge[0]]["op"]) is ops.CNOT
               and type(circuit.dag.nodes[edge[1]]["op"]) is not ops.OneQubitGateWrapper
        ]

        if len(edges) == 0:
            self.replace_photon_one_qubit_op(circuit)
        else:

            # select relevant register and location of the gate to be inserted
            ind = np.random.randint(len(edges))
            edge = list(edges)[ind]
            reg = circuit.dag.edges[edge]["reg"]

            # select a random local Clifford gate
            ind = np.random.choice(len(self.one_qubit_ops), p=self.p_dist)
            op = self.one_qubit_ops[ind]
            noise = self._wrap_noise(op, self.noise_model_mapping["p"])
            gate = ops.OneQubitGateWrapper(op, reg_type="p", register=reg, noise=noise)
            circuit.insert_at(gate, [edge])

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
            self.replace_emitter_one_qubit_op(circuit)
        else:
            # select relevant register and location of the gate to be inserted
            ind = np.random.randint(len(edges))
            edge = list(edges)[ind]
            reg = circuit.dag.edges[edge]["reg"]

            # select a random local Clifford gate
            ind = np.random.choice(len(self.one_qubit_ops), p=self.e_dist)
            op = self.one_qubit_ops[ind]
            noise = self._wrap_noise(op, self.noise_model_mapping["e"])
            gate = ops.OneQubitGateWrapper(op, reg_type="e", register=reg, noise=noise)
            circuit.insert_at(gate, [edge])

    def add_emitter_cnot(self, circuit):
        """
        Randomly selects two valid edges on which to add a new two-qubit gate. One edge is selected from all emitter
        edges, and then the second is selected that maintains proper temporal ordering of the operations.

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
            noise=self._identify_noise(ops.CNOT, self.noise_model_mapping["ee"]),
        )

        circuit.insert_at(gate, [edge0, edge1])

    def remove_op(self, circuit, node=None):
        """
        Remove an operation in CircuitDAG.
        The node to be removed is either selected by a user or chosen randomly subject to some restrictions

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
        else:
            nodes = circuit.get_node_by_labels(["Fixed"])
            if node in nodes:
                return

        circuit.remove_op(node)

    def add_measurement_cnot_and_reset(self, circuit):
        """
        Add a MeasurementCNOTandReset operation from an emitter qubit to a photonic qubit such that no consecutive
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
                ops.MeasurementCNOTandReset, self.noise_model_mapping["ep"]
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
            "solver setting": self.setting,
            "seed": self.last_seed,
            "One-qubit ops": op_names(self.one_qubit_ops),
            "Transition probabilities": transition_names(self.trans_probs),
        }
