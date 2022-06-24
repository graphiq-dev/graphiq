import matplotlib.pyplot as plt

import numpy as np
import networkx as nx
import warnings
import time

import src.backends.density_matrix.functions as dmf

from src.metrics import MetricBase
from src.circuit import CircuitBase
from src.backends.compiler_base import CompilerBase

from src.solvers.random_solver import RandomSearchSolver

from src.backends.density_matrix.compiler import DensityMatrixCompiler
from src.circuit import CircuitDAG
from src.metrics import Infidelity

from src.visualizers.density_matrix import density_matrix_bars
from src import ops

import benchmarks.circuits as bc


class RuleBasedRandomSearchSolver(RandomSearchSolver):
    """
    Implements a rule-based random search solver.
    This will randomly add/delete/modify operations in the circuit.
    """

    name = "rule-based-random-search"
    n_stop = 50  # maximum number of iterations
    n_pop = 50
    n_hof = 5
    tournament_k = 2  # tournament size for selection of the next population

    fixed_ops = [  # ops that should never be removed/swapped
        ops.Input,
        ops.Output
    ]

    single_qubit_ops = list(
        ops.single_qubit_cliffords()
    )

    def __init__(self, target, metric: MetricBase, compiler: CompilerBase, circuit: CircuitBase = None,
                 n_emitter=1, n_photon=1, selection_active=False, *args, **kwargs):

        super().__init__(target, metric, compiler, circuit, *args, **kwargs)

        # update class variables, e.g., n_stop, h_pop, by passing in kwargs
        self.__dict__.update(kwargs)

        self.n_emitter = n_emitter
        self.n_photon = n_photon

        # transformation functions and their relative probabilities
        self.trans_probs = self.initialize_transformation_probabilities()
        self.selection_active = selection_active

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
        """
        self.trans_probs[self.add_emitter_one_qubit_op] += -1 / self.n_stop / 3
        self.trans_probs[self.replace_photon_one_qubit_op] += -1 / self.n_stop / 3
        self.trans_probs[self.remove_op] += 1 / self.n_stop
        if self.n_emitter > 1:
            self.trans_probs[self.add_emitter_cnot] += -1 / self.n_stop / 3

        # normalize the probabilities
        total = np.sum(list(self.trans_probs.values()))
        for key in self.trans_probs.keys():
            self.trans_probs[key] *= 1 / total


    @staticmethod
    def _initialization(emission_assignment, measurement_assignment):
        """
        Initialize a quantum circuit with photon emission, emitter measurements

        :param emission_assignment: which emitter emits which photon
        :type emission_assignment: list[int]
        :param measurement_assignment: which photonic qubit is targeted after measuring each emitter
        :type measurement_assignment: list[int]
        :return: nothing
        """
        n_photon = len(emission_assignment)
        n_emitter = len(measurement_assignment)
        circuit = CircuitDAG(n_emitter=n_emitter, n_photon=n_photon, n_classical=1)

        for i in range(n_photon):
            # initialize all photon emission gates
            circuit.add(ops.CNOT(control=emission_assignment[i], control_type='e', target=i, target_type='p'))
            # initialize all single-qubit Clifford gate for photonic qubits
            circuit.add(ops.SingleQubitGateWrapper([ops.Identity, ops.Hadamard], register=i, reg_type='p'))

        # initialize all emitter meausurement and reset operations

        for j in range(n_emitter):
            circuit.add(
                ops.MeasurementCNOTandReset(control=j, control_type='e', target=measurement_assignment[j],
                                            target_type='p'))
        return circuit

    def test_initialization(self):
        # debugging only
        emission_assignment = self.get_emission_assignment(self.n_photon, self.n_emitter)
        measurement_assignment = self.get_measurement_assignment(self.n_photon, self.n_emitter)
        circuit = self._initialization(emission_assignment, measurement_assignment)
        circuit.draw_dag()
        circuit.draw_circuit()

    def test_more_measurements(self):
        # debugging only
        emission_assignment = self.get_emission_assignment(self.n_photon, self.n_emitter)
        measurement_assignment = self.get_measurement_assignment(self.n_photon, self.n_emitter)
        circuit = self._initialization(emission_assignment, measurement_assignment)
        self.add_measurement_cnot_and_reset(circuit)
        self.add_measurement_cnot_and_reset(circuit)
        circuit.draw_dag()
        circuit.draw_circuit()

    def solve(self):
        """
        The main function for the solver

        :return: function returns nothing
        :rtype: None
        """

        # TODO: add some logging to see how well it performed at each epoch (and pick n_stop accordingly)

        # Initialize population
        population = []
        for j in range(self.n_pop):
            emission_assignment = self.get_emission_assignment(self.n_photon, self.n_emitter)
            measurement_assignment = self.get_measurement_assignment(self.n_photon, self.n_emitter)

            circuit = self._initialization(emission_assignment, measurement_assignment)

            fixed_node = circuit.dag.nodes()
            population.append((np.inf, circuit))  # initialize all population members

        for i in range(self.n_stop):
            for j in range(self.n_pop):
                transformation = np.random.choice(list(self.trans_probs.keys()), p=list(self.trans_probs.values()))
                circuit = population[j][1]

                if transformation == self.add_emitter_one_qubit_op:
                    self.add_emitter_one_qubit_op(circuit, self.e_dist)

                elif transformation == self.add_photon_one_qubit_op:
                    self.add_photon_one_qubit_op(circuit, self.p_dist)

                elif transformation == self.replace_photon_one_qubit_op:
                    self.replace_photon_one_qubit_op(circuit, self.p_dist)

                elif transformation == self.remove_op:
                    self.remove_op(circuit, fixed_node)

                else:
                    transformation(circuit)

                circuit.validate()

                compiled_state = self.compiler.compile(circuit)  # this will pass out a density matrix object

                state_data = dmf.partial_trace(compiled_state.data,
                                               keep=list(range(self.n_photon)),
                                               dims=(self.n_photon + self.n_emitter) * [2])
                score = self.metric.evaluate(state_data, circuit)

                population[j] = (score, circuit)

            self.update_hof(population)
            # self.adapt_probabilities(i)
            if self.selection_active:
                population = self.tournament_selection(population, k=self.tournament_k)

            print(f"Iteration {i} | Best score: {self.hof[0][0]:.4f}")

    def replace_photon_one_qubit_op(self, circuit, p_dist):
        """
        Replace one single-qubit Clifford gate applied on a photonic qubit to another one.

        :param circuit: a quantum circuit
        :type circuit: CircuitDAG
        :param p_dist: probability distribution
        :type p_dist: list[float] or list[double]
        :return: nothing
        """

        nodes = [node for node in circuit.dag.nodes
                 if type(circuit.dag.nodes[node]['op']) is ops.SingleQubitGateWrapper and
                 list(circuit.dag.in_edges(nbunch=node, data=True))[0][2]['reg_type'] == 'p']
        if len(nodes) == 0:
            return
        ind = np.random.randint(len(nodes))
        node = list(nodes)[ind]

        old_op = circuit.dag.nodes[node]['op']

        reg = old_op.register
        ind = np.random.choice(len(self.single_qubit_ops), p=p_dist)
        op = self.single_qubit_ops[ind]
        gate = ops.SingleQubitGateWrapper(op, reg_type='p', register=reg)
        circuit._open_qasm_update(gate)
        # TODO: check that the gate is guaranteed to be in the original position
        circuit.dag.nodes[node]['op'] = gate

    def add_photon_one_qubit_op(self, circuit, p_dist):
        """
        Add a single-qubit Clifford gate to a photonic qubit
        Will be removed if replacing and initialization work.

        :param circuit: a quantum circuit
        :type circuit: CircuitDAG
        :param p_dist: probability distribution
        :type p_dist: list[float] or list[double]
        :return: function returns nothing
        :rtype: None
        """
        # make sure only selecting a photonic qubit after initialization and avoiding applying single-qubit Clifford gates twice
        edges = [edge for edge in circuit.dag.edges if circuit.dag.edges[edge]['reg_type'] == 'p' and
                 type(circuit.dag.nodes[edge[0]]['op']) is ops.CNOT and
                 type(circuit.dag.nodes[edge[1]]['op']) is not ops.SingleQubitGateWrapper]

        if len(edges) == 0:
            return

        ind = np.random.randint(len(edges))
        edge = list(edges)[ind]

        reg = circuit.dag.edges[edge]['reg']
        label = edge[2]

        new_id = circuit._unique_node_id()
        new_edges = [(edge[0], new_id, label), (new_id, edge[1], label)]

        ind = np.random.choice(len(self.single_qubit_ops), p=p_dist)
        op = self.single_qubit_ops[ind]
        gate = ops.SingleQubitGateWrapper(op, reg_type='p', register=reg)
        circuit._open_qasm_update(gate)

        circuit.dag.add_node(new_id, op=gate)
        circuit.dag.remove_edges_from([edge])  # remove the edge

        circuit.dag.add_edges_from(new_edges, reg_type='p', reg=reg)

    def add_emitter_one_qubit_op(self, circuit, e_dist):
        """
        Randomly selects one valid edge on which to add a new single-qubit gate

        :param circuit: a quantum circuit
        :type circuit: CircuitDAG
        :param e_dist: probability distribution
        :type e_dist: list[float] or list[double]
        :return: nothing
        """
        # make sure only selecting emitter qubits and avoiding adding a gate after final measurement or
        # adding two single-qubit Clifford gates in a row
        edges = [edge for edge in circuit.dag.edges if circuit.dag.edges[edge]['reg_type'] == 'e' and
                 type(circuit.dag.nodes[edge[1]]['op']) is not ops.Output and
                 type(circuit.dag.nodes[edge[0]]['op']) is not ops.SingleQubitGateWrapper and
                 type(circuit.dag.nodes[edge[1]]['op']) is not ops.SingleQubitGateWrapper]

        if len(edges) == 0:
            return

        ind = np.random.randint(len(edges))
        edge = list(edges)[ind]

        reg = circuit.dag.edges[edge]['reg']
        label = edge[2]

        new_id = circuit._unique_node_id()
        new_edges = [(edge[0], new_id, label), (new_id, edge[1], label)]

        ind = np.random.choice(len(self.single_qubit_ops), p=e_dist)
        op = self.single_qubit_ops[ind]
        gate = ops.SingleQubitGateWrapper(op, reg_type='e', register=reg)
        circuit._open_qasm_update(gate)

        circuit.dag.add_node(new_id, op=gate)
        circuit.dag.remove_edges_from([edge])  # remove the edge

        circuit.dag.add_edges_from(new_edges, reg_type='e', reg=reg)

    def add_emitter_cnot(self, circuit):
        """
        Randomly selects two valid edges on which to add a new two-qubit gate.
        One edge is selected from all edges, and then the second is selected that maintains proper temporal ordering
        of the operations.

        :param circuit: a quantum circuit
        :type circuit: CircuitDAG
        :return: nothing
        """
        possible_edge_pairs = self._select_possible_cnot_position(circuit)
        if len(possible_edge_pairs) == 0:
            warnings.warn("No valid registers to place the two-qubit gate")
            return

        ind = np.random.randint(len(possible_edge_pairs))
        (edge0, edge1) = possible_edge_pairs[ind]

        new_id = circuit._unique_node_id()
        gate = ops.CNOT(control=circuit.dag.edges[edge0]['reg'], control_type='e',
                        target=circuit.dag.edges[edge1]['reg'], target_type='e')
        circuit._open_qasm_update(gate)
        circuit.dag.add_node(new_id, op=gate)

        for edge in [edge0, edge1]:
            reg = circuit.dag.edges[edge]['reg']
            label = edge[2]
            new_edges = [(edge[0], new_id, label), (new_id, edge[1], label)]

            circuit.dag.add_edges_from(new_edges, reg_type='e', reg=reg)

        circuit.dag.remove_edges_from([edge0, edge1])  # remove the selected edges

    def remove_op(self, circuit, fixed_node, node=None):
        """
        Randomly selects a node in CircuitDAG to remove subject to some restrictions

        :param circuit: a quantum circuit
        :type circuit: CircuitDAG
        :param fixed_node: a list of nodes in CircuitDAG that should not be modified
        :type fixed_node: list[int]
        :param node: a specified node (by node_id) to be removed
        :type node: int
        :return: nothing
        """
        if node is None:
            nodes = [node for node in circuit.dag.nodes if not (
                    (type(circuit.dag.nodes[node]['op']) in self.fixed_ops) or (node in fixed_node))]

            if len(nodes) == 0:
                return

            ind = np.random.randint(len(nodes))
            node = nodes[ind]

        in_edges = list(circuit.dag.in_edges(node, keys=True))
        out_edges = list(circuit.dag.out_edges(node, keys=True))

        for in_edge in in_edges:
            for out_edge in out_edges:
                if in_edge[2] == out_edge[2]:  # i.e. if the keys are the same
                    reg = circuit.dag.edges[in_edge]['reg']
                    reg_type = circuit.dag.edges[in_edge]['reg_type']
                    label = out_edge[2]
                    circuit.dag.add_edge(in_edge[0], out_edge[1], label, reg_type=reg_type, reg=reg)

        circuit.dag.remove_node(node)

    def add_measurement_cnot_and_reset(self, circuit):
        """
        Add a MeausrementCNOTandReset operation from an emitter qubit to a photonic qubit such that no consecutive MeasurementCNOTReset is allowed.
        This operation cannot be added before the photonic qubit is initialized.

        :param circuit: a quantum circuit
        :type circuit: CircuitDAG
        return: nothing
        """

        possible_edge_pairs = self._select_possible_measurement_position(circuit)

        if len(possible_edge_pairs) == 0:
            warnings.warn("No valid registers to place the two-qubit gate")
            return

        ind = np.random.randint(len(possible_edge_pairs))
        (edge0, edge1) = possible_edge_pairs[ind]

        new_id = circuit._unique_node_id()
        gate = ops.MeasurementCNOTandReset(control=circuit.dag.edges[edge0]['reg'], control_type='e',
                                           target=circuit.dag.edges[edge1]['reg'], target_type='p')

        circuit._open_qasm_update(gate)
        circuit.dag.add_node(new_id, op=gate)

        reg = circuit.dag.edges[edge0]['reg']
        label = edge0[2]
        new_edges = [(edge0[0], new_id, label), (new_id, edge0[1], label)]

        circuit.dag.add_edges_from(new_edges, reg_type='e', reg=reg)

        reg = circuit.dag.edges[edge1]['reg']
        label = edge1[2]
        new_edges = [(edge1[0], new_id, label), (new_id, edge1[1], label)]

        circuit.dag.add_edges_from(new_edges, reg_type='p', reg=reg)

        circuit.dag.remove_edges_from([edge0, edge1])  # remove the selected edges

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
        edges = [edge for edge in circuit.dag.edges if
                 circuit.dag.edges[edge]['reg_type'] == 'e' and type(
                     circuit.dag.nodes[edge[1]]['op']) is not ops.Output]

        for edge in edges:

            ancestors = nx.ancestors(circuit.dag, edge[0])
            descendants = nx.descendants(circuit.dag, edge[1])

            ancestor_edges = list(circuit.dag.in_edges(edge[0], keys=True))
            for anc in ancestors:
                ancestor_edges.extend(circuit.dag.edges(anc, keys=True))

            descendant_edges = list(circuit.dag.out_edges(edge[1], keys=True))
            for des in descendants:
                descendant_edges.extend(circuit.dag.edges(des, keys=True))

            possible_edges = set(edges) - set([edge]) - set(ancestor_edges) - set(descendant_edges)

            for another_edge in possible_edges:
                edge_pair.append((edge, another_edge))

        return edge_pair

    @staticmethod
    def _select_possible_measurement_position(circuit):
        """
        Find all possible positions to add the MeasurementCNOTandReset operation

        :param circuit: a quantum circuit
        :type circuit: CircuitDAG
        :return: return a list of edge pairs
        :rtype:list[tuple]
        """
        edge_pair = []
        e_edges = [edge for edge in circuit.dag.edges if
                   circuit.dag.edges[edge]['reg_type'] == 'e' and type(
                       circuit.dag.nodes[edge[1]]['op']) is not (ops.Output and ops.MeasurementCNOTandReset)
                   and type(circuit.dag.nodes[edge[0]]['op']) is not ops.MeasurementCNOTandReset]

        p_edges = [edge for edge in circuit.dag.edges if
                   circuit.dag.edges[edge]['reg_type'] == 'p' and type(
                       circuit.dag.nodes[edge[1]]['op']) is not ops.MeasurementCNOTandReset
                   and type(circuit.dag.nodes[edge[0]]['op']) is not ops.Input]

        for edge in e_edges:
            ancestors = nx.ancestors(circuit.dag, edge[0])
            descendants = nx.descendants(circuit.dag, edge[1])
            ancestor_edges = [edge for edge in circuit.dag.in_edges(edge[0], keys=True) if
                              circuit.dag.edges[edge]['reg_type'] == 'p']
            for anc in ancestors:
                ancestor_edges.extend(circuit.dag.edges(anc, keys=True))

            descendant_edges = [edge for edge in circuit.dag.out_edges(edge[1], keys=True) if
                                circuit.dag.edges[edge]['reg_type'] == 'p']
            for des in descendants:
                descendant_edges.extend(circuit.dag.edges(des, keys=True))
            possible_edges = set(p_edges) - set(ancestor_edges) - set(descendant_edges)

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
            'solver name': self.name,
            'Max iteration #': self.n_stop,
            'Population size': self.n_pop,
            'seed': self.last_seed,
            'Fixed ops': op_names(self.fixed_ops),
            'Single qubit ops': op_names(self.single_qubit_ops),
            'Transition probabilities': transition_names(self.trans_probs)
        }


if __name__ == "__main__":
    #%% here we have access
    RuleBasedRandomSearchSolver.n_stop = 40
    RuleBasedRandomSearchSolver.n_pop = 150
    RuleBasedRandomSearchSolver.n_hof = 10
    RuleBasedRandomSearchSolver.tournament_k = 10

    #%% comment/uncomment for reproducibility
    # RuleBasedRandomSearchSolver.seed(1)

    # %% select which state we want to target
    from benchmarks.circuits import *
    circuit_ideal, state_ideal = linear_cluster_3qubit_circuit()

    #%% construct all of our important objects
    target = state_ideal['dm']
    compiler = DensityMatrixCompiler()
    metric = Infidelity(target=target)

    solver = RuleBasedRandomSearchSolver(target=target, metric=metric, compiler=compiler)

    #%% call the solver.solve() function to implement the random search algorithm
    t0 = time.time()
    solver.solve()
    t1 = time.time()

    #%% print/plot the results
    print(solver.hof)
    print(f"Total time {t1-t0}")

    circuit = solver.hof[0][1]

    # extract the best performing circuit
    fig, axs = density_matrix_bars(target)
    fig.suptitle("Target density matrix")
    plt.show()

    state = compiler.compile(circuit)
    fig, axs = density_matrix_bars(state.data)
    fig.suptitle("Simulated density matrix")
    plt.show()

    circuit.draw_circuit()
