import matplotlib.pyplot as plt

import numpy as np
import networkx as nx
import warnings
import collections
import copy
import time

import src.backends.density_matrix.functions as dmf

from src.solvers.base import SolverBase
# from src.metrics import MetricBase
# from src.circuit import CircuitBase
# from src.backends.compiler_base import CompilerBase

from src.backends.density_matrix.compiler import DensityMatrixCompiler
from src.circuit import CircuitDAG
from src.metrics import Infidelity

from src.visualizers.density_matrix import density_matrix_bars
from src import ops
from src.io import IO

import benchmarks.circuits as bc


class RuleBasedRandomSearchSolver(SolverBase):
    """
    Implements a rule-based random search solver.
    This will randomly add/delete operations in the circuit.
    """

    name = "rule-based-random-search"
    n_stop = 200  # maximum number of iterations
    n_hof = 10
    n_pop = 50

    fixed_ops = [  # ops that should never be removed/swapped
        ops.Input,
        ops.Output
    ]
    single_qubit_ops = list(ops.single_qubit_cliffords())

    def __init__(self,
                 target=None,
                 metric=None,
                 circuit=None,
                 compiler=None,
                 n_emitter=1,
                 n_photon=1,
                 *args, **kwargs):
        super().__init__(target, metric, circuit, compiler, *args, **kwargs)
        self.n_emitter = n_emitter
        self.n_photon = n_photon
        if self.n_emitter > 1:
            self.transformations = [
                self.add_emitter_one_qubit_op,
                self.add_emitter_cnot,
                #self.add_photon_one_qubit_op,
                self.replace_photon_one_qubit_op,
                self.remove_op
            ]
            self.transformation_prob = [0.4, 0.1, 0.1, 0.4]
        else:
            self.transformations = [
                self.add_emitter_one_qubit_op,
                #self.add_photon_one_qubit_op,
                self.replace_photon_one_qubit_op,
                self.remove_op,
                # self.add_measurement_and_reset
            ]
            self.transformation_prob = [1 / 3, 1 / 3, 1 / 3]
            # self.transformation_prob = [1 / 2, 1 / 2]
        # hof stores the best circuits and their scores in the form of: (scores, circuits)
        self.hof = (collections.deque(self.n_hof * [np.inf]),
                    collections.deque(self.n_hof * [None]))

    @staticmethod
    def _initialization(n_emitter, n_photon, emission_assignment, measurement_assignment):
        """
        Initialize a quantum circuit with photon emission, emitter measurements
        :param n_emitter: number of emitters
        :type n_emitter: int
        :param n_photon: number of photonic qubits
        :type n_photon: int
        :param emission_assignment: which emitter emits which photon
        :type emission_assignment: list[int]
        :param measurement_assignment: which photonic qubit is targeted after measuring each emitter
        :type measurement_assignment: list[int]
        :return: nothing
        """
        circuit = CircuitDAG(n_emitter=n_emitter, n_photon=n_photon, n_classical=1)
        assert len(emission_assignment) == n_photon

        for i in range(n_photon):
            # initialize all photon emission gates
            circuit.add(ops.CNOT(control=emission_assignment[i], control_type='e', target=i, target_type='p'))
            # initialize all single-qubit Clifford gate for photonic qubits
            circuit.add(ops.SingleQubitGateWrapper([ops.Identity, ops.Hadamard], register=i, reg_type='p'))

        # initialize all emitter meausurement and reset operations
        assert len(measurement_assignment) == n_emitter
        for j in range(n_emitter):
            circuit.add(
                ops.MeasurementCNOTandReset(control=j, control_type='e', target=measurement_assignment[j], target_type='p'))
        return circuit

    def update_hof(self, score, circuit):
        for i in range(self.n_hof):
            if score < self.hof[0][i]:
                self.hof[0].insert(i, copy.deepcopy(score))
                self.hof[1].insert(i, copy.deepcopy(circuit))

                self.hof[0].pop()
                self.hof[1].pop()
                break

    def test_initialization(self, seed):
        # debugging only
        np.random.seed(seed)

        emission_assignment = RuleBasedRandomSearchSolver.get_emission_assignment(self.n_photon, self.n_emitter)
        measurement_assignment = RuleBasedRandomSearchSolver.get_measurement_assignment(self.n_photon, self.n_emitter)
        circuit = self._initialization(self.n_emitter, self.n_photon, emission_assignment, measurement_assignment)
        circuit.draw_dag()
        # print(circuit.dag.nodes())
        circuit.draw_circuit()

    def solve(self, seed):
        """
        The main function for the solver
        :param seed: a random number generator seed
        :type seed: int
        :return: function returns nothing
        :rtype: None
        """
        np.random.seed(seed)
        p_dist = [0.5] + 11 * [0.1 / 22] + [0.4] + 11 * [0.1 / 22]
        e_dist = [0.5] + 11 * [0.02 / 22] + [0.48] + 11 * [0.02 / 22]

        # Initialize population
        circuit_pop = [None] * self.n_pop
        for i in range(self.n_pop):
            emission_assignment = RuleBasedRandomSearchSolver.get_emission_assignment(self.n_photon,
                                                                                      self.n_emitter)
            measurement_assignment = RuleBasedRandomSearchSolver.get_measurement_assignment(self.n_photon,
                                                                                            self.n_emitter)
            circuit = self._initialization(self.n_emitter, self.n_photon, emission_assignment, measurement_assignment)
            fixed_node = circuit.dag.nodes()
            circuit_pop[i] = circuit
            # print(f"\nNew generation {i}")

        for _ in range(self.n_stop):
            for i in range(self.n_pop):
                transformation = self.transformations[
                    np.random.choice(len(self.transformations), p=self.transformation_prob)]
                # transformation = self.add_emitter_CNOT
                # transformation = self.add_emitter_one_qubit_op

                if transformation == self.add_emitter_one_qubit_op:

                    self.add_emitter_one_qubit_op(circuit_pop[i], e_dist)

                elif transformation == self.add_photon_one_qubit_op:
                    self.add_photon_one_qubit_op(circuit_pop[i], p_dist)

                elif transformation == self.replace_photon_one_qubit_op:
                    self.replace_photon_one_qubit_op(circuit_pop[i], p_dist)

                elif transformation == self.remove_op:
                    self.remove_op(circuit_pop[i], fixed_node)
                else:
                    transformation(circuit_pop[i])

                circuit_pop[i].validate()

                compiled_state = self.compiler.compile(circuit_pop[i])  # this will pass out a density matrix object

                state_data = dmf.partial_trace(compiled_state.data, keep=list(range(self.n_photon)),
                                               dims=(self.n_photon + self.n_emitter) * [2])
                score = self.metric.evaluate(state_data, circuit_pop[i])

                self.update_hof(score, circuit_pop[i])

    def replace_photon_one_qubit_op(self, circuit, p_dist):
        """
        Replacing one single-qubit Clifford gate applied on a photonic qubit to another one
        :param circuit: a quantum circuit
        :type circuit: CircuitDAG
        :param p_dist: probability distribution
        :type p_dist: list[float] or list[double]
        :return:
        """
        # TODO: debug this function
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
        # TODO: fix an issue to place the gate in the original position
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
        edges = [edge for edge in circuit.dag.edges if
                 circuit.dag.edges[edge]['reg_type'] == 'e' and type(circuit.dag.nodes[edge[1]]['op']) is not ops.Output]

        ind = np.random.randint(len(edges))
        edge0 = list(edges)[ind]

        ancestors = nx.ancestors(circuit.dag, edge0[0])
        descendants = nx.descendants(circuit.dag, edge0[1])

        ancestor_edges = list(circuit.dag.in_edges(edge0[0], keys=True))
        for anc in ancestors:
            ancestor_edges.extend(circuit.dag.edges(anc, keys=True))

        descendant_edges = list(circuit.dag.out_edges(edge0[1], keys=True))
        for des in descendants:
            descendant_edges.extend(circuit.dag.edges(des, keys=True))

        possible_edges = set(edges) - set([edge0]) - set(ancestor_edges) - set(descendant_edges)
        if len(possible_edges) == 0:
            warnings.warn("No valid registers to place the two-qubit gate")
            return

        ind = np.random.randint(len(possible_edges))
        edge1 = list(possible_edges)[ind]

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
        :type fixed_node:
        :param node: a specified node to be removed
        :type node:
        :return: nothing
        """
        if node is None:
            nodes = [node for node in circuit.dag.nodes if not (
                    not (type(circuit.dag.nodes[node]['op']) not in self.fixed_ops) or not (
                    node not in fixed_node))]

            if len(nodes) == 0:
                return

            ind = np.random.randint(len(nodes))
            node = nodes[ind]

        in_edges = list(circuit.dag.in_edges(node, keys=True))
        out_edges = list(circuit.dag.out_edges(node, keys=True))

        for in_edge in in_edges:
            for out_edge in out_edges:
                if in_edge[2] == out_edge[2]:
                    reg = circuit.dag.edges[in_edge]['reg']
                    label = out_edge[2]
                    circuit.dag.add_edge(in_edge[0], out_edge[1], label, reg_type='e', reg=reg)

        circuit.dag.remove_node(node)

    # helper functions
    @staticmethod
    def get_emission_assignment(n_photon, n_emitter):
        """
        Generate a random assignment for the emission source of each photon
        :param n_photon: number of photons
        :type n_photon: int
        :param n_emitter: number of emitters
        :type n_emitter: int
        :param seed: a random number generator seed
        :type seed: int
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
        :param seed: a random number generator seed
        :type seed: int
        :return: a list of photon numbers
        :rtype: list[int]
        """
        return np.random.randint(n_photon, size=n_emitter).tolist()


    return np.random.randint(n_photon, size=n_emitter).tolist()


if __name__ == "__main__":
    circuit_ideal, state_ideal = bc.linear_cluster_3qubit_circuit()
    target = state_ideal['dm']
    n_photon = 3
    n_emitter = 1
    compiler = DensityMatrixCompiler()
    metric = MetricFidelity(target=target)

    solver = RuleBasedRandomSearchSolver(target=target, metric=metric, compiler=compiler, n_emitter=n_emitter, n_photon=n_photon)
    solver.solve(200)
    print('hof score is ' + str(solver.hof[0][0]))
    circuit = solver.hof[1][0]
    state = compiler.compile(circuit)
    state2 = compiler.compile(circuit)
    state3 = compiler.compile(circuit)
    circuit.draw_circuit()
    # circuit.draw_dag()
    fig, axs = density_matrix_bars(target)
    fig.suptitle("TARGET DENSITY MATRIX")
    plt.show()

    new_state = dmf.partial_trace(state.data, keep=list(range(n_photon)), dims=(n_photon + n_emitter) * [2])
    new_state2 = dmf.partial_trace(state2.data, keep=list(range(n_photon)), dims=(n_photon + n_emitter) * [2])
    new_state3 = dmf.partial_trace(state3.data, keep=list(range(n_photon)), dims=(n_photon + n_emitter) * [2])
    print('Are these two states the same: ' + str(np.allclose(new_state, new_state3)))
    print('The circuit compiles a state that has an infidelity ' + str(metric.evaluate(new_state, circuit)))
    fig, axs = density_matrix_bars(new_state)

    fig.suptitle("CREATED DENSITY MATRIX")
    plt.show()
