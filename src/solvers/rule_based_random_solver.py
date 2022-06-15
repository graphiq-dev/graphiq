import matplotlib.pyplot as plt

import numpy as np
import networkx as nx
import warnings
import collections
import copy
import time

import src.backends.density_matrix.functions as dmf

from src.solvers.base import SolverBase
from src.metrics import MetricBase
from src.circuit import CircuitBase
from src.backends.compiler_base import CompilerBase

from src.backends.density_matrix.compiler import DensityMatrixCompiler
from src.circuit import CircuitDAG
from src.metrics import MetricFidelity

from src.visualizers.density_matrix import density_matrix_bars
from src import ops
from src.io import IO

from benchmarks.circuits import *


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
    single_qubit_ops = list(single_qubit_cliffords())

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
                                self.add_emitter_CNOT,
                                self.add_photon_one_qubit_op,
                                self.remove_op
                                ]
            self.transformation_prob = [0.4, 0.1, 0.1, 0.4]
        else:
            self.transformations = [
                                self.add_emitter_one_qubit_op,
                                self.add_photon_one_qubit_op,
                                self.remove_op
                                ]
            self.transformation_prob = [1/3, 1/3, 1/3]
        # hof stores the best circuits and their scores in the form of: (scores, circuits)
        self.hof = (collections.deque(self.n_hof * [np.inf]),
                    collections.deque(self.n_hof * [None]))

    def _initialization(self, n_emitter, n_photon, emission_assignment, measurement_assignment):
        circuit = CircuitDAG(n_emitter=n_emitter, n_photon=n_photon, n_classical=0)
        assert len(emission_assignment) == n_photon

        # initialize all photon emission gates
        for i in range(n_photon):
            circuit.add(CNOT(control=emission_assignment[i], control_type='e', target=i, target_type='p'))
        #for i in range(n_photon):
         #   circuit.add(SingleQubitGateWrapper([Identity, Hadamard], register=i, reg_type='p'))
        # initialize all emitter meausurement and reset operations

        assert len(measurement_assignment) == n_emitter
        for j in range(n_emitter):
            circuit.add(
                MeasurementCNOTandReset(control=j, control_type='e', target=measurement_assignment[j], target_type='p'))
        return circuit

    def update_hof(self, scores, circuits):
        for score, circuit in zip(scores, circuits):
            for i in range(self.n_hof):
                if score < self.hof[0][i]:
                    self.hof[0].insert(i, copy.deepcopy(score))
                    self.hof[1].insert(i, copy.deepcopy(circuit))

                    self.hof[0].pop()
                    self.hof[1].pop()
                    break

    def test_initialization(self, seed):
        np.random.seed(seed)

        emission_assignment = get_emission_assignment(self.n_photon, self.n_emitter, np.random.randint(10000))
        measurement_assignment = get_measurement_assignment(self.n_photon, self.n_emitter,
                                                                np.random.randint(10000))
        circuit = self._initialization(self.n_emitter, self.n_photon, emission_assignment, measurement_assignment)
        circuit.draw_dag()
        #print(circuit.dag.nodes())
        circuit.draw_circuit()

    def solve(self, seed):

        scores = [None for _ in range(self.n_stop)]
        circuits = [None for _ in range(self.n_stop)]
        np.random.seed(seed)
        p_dist = [0.5] + 11 * [0.1/22] + [0.4] + 11 * [0.1/22]
        e_dist = [0.5] + 11 * [0.02/22] + [0.48] + 11 * [0.02/22]
        for i in range(self.n_pop):
            emission_assignment = get_emission_assignment(self.n_photon, self.n_emitter, np.random.randint(10000))
            measurement_assignment = get_measurement_assignment(self.n_photon, self.n_emitter, np.random.randint(10000))
            circuit = self._initialization(self.n_emitter, self.n_photon, emission_assignment, measurement_assignment)
            fixed_node = circuit.dag.nodes()
            # print(f"\nNew generation {i}")

            for j in range(self.n_stop):
                transformation = self.transformations[np.random.choice(len(self.transformations), p=self.transformation_prob)]
                # transformation = self.add_emitter_CNOT
                # transformation = self.add_emitter_one_qubit_op

                if transformation == self.add_emitter_one_qubit_op:

                    self.add_emitter_one_qubit_op(circuit, e_dist)

                elif transformation == self.add_photon_one_qubit_op:
                    self.add_photon_one_qubit_op(circuit, p_dist)

                elif transformation == self.replace_photon_one_qubit_op:
                    self.replace_photon_one_qubit_op(circuit, p_dist)

                elif transformation == self.remove_op:
                    self.remove_op(circuit, fixed_node)
                else:
                    transformation(circuit)


                circuit.validate()

                state = self.compiler.compile(circuit)  # this will pass out a density matrix object

                state = dmf.partial_trace(state.data, list(range(self.n_photon)), (self.n_photon + self.n_emitter) * [2])
                # score = 1
                score = self.metric.evaluate(state.data, circuit)

                # print(score)

                scores[j] = score
                circuits[j] = circuit

            self.update_hof(scores, circuits)

        return

    def replace_photon_one_qubit_op(self, circuit, p_dist):
        # TODO: debug this function
        nodes = [node for node in circuit.dag.nodes if type(circuit.dag.nodes[node]['op']) is SingleQubitGateWrapper]
        if len(nodes) == 0:
            return
        ind = np.random.randint(len(nodes))
        node = list(nodes)[ind]

        old_op = circuit.dag.nodes[node]['op']
        print(old_op.register)
        reg = old_op.register
        op = np.random.choice(self.single_qubit_ops, p=p_dist)
        gate = SingleQubitGateWrapper(op, reg_type='p', register=reg)
        circuit._open_qasm_update(gate)
        # TODO: fix an issue to place the gate in the original position
        circuit.dag.nodes[node]['op'] = gate

    def add_photon_one_qubit_op(self, circuit, p_dist):
        edges = [edge for edge in circuit.dag.edges if circuit.dag.edges[edge]['reg_type'] == 'p' and
                                                    type(circuit.dag.nodes[edge[0]]['op']) is CNOT and
                                                    type(circuit.dag.nodes[edge[1]]['op']) is not SingleQubitGateWrapper]

        if len(edges) == 0:
            return
        ind = np.random.randint(len(edges))
        edge = list(edges)[ind]

        reg = circuit.dag.edges[edge]['reg']
        label = edge[2]

        new_id = circuit._unique_node_id()
        new_edges = [(edge[0], new_id, label), (new_id, edge[1], label)]

        op = np.random.choice(self.single_qubit_ops, p=p_dist)
        gate = SingleQubitGateWrapper(op, reg_type='p', register=reg)
        circuit._open_qasm_update(gate)

        circuit.dag.add_node(new_id, op=gate)
        circuit.dag.remove_edges_from([edge])  # remove the edge

        circuit.dag.add_edges_from(new_edges, reg_type='p', reg=reg)
        return


    def add_emitter_one_qubit_op(self, circuit, e_dist):
        """
        Randomly selects one valid edge on which to add a new single-qubit gate
        """

        edges = [edge for edge in circuit.dag.edges if circuit.dag.edges[edge]['reg_type'] == 'e' and
                                                    type(circuit.dag.nodes[edge[1]]['op']) is not Output and
                                                    type(circuit.dag.nodes[edge[0]]['op']) is not SingleQubitGateWrapper and
                                                    type(circuit.dag.nodes[edge[1]]['op']) is not SingleQubitGateWrapper]

        if len(edges) ==0:
            return
        ind = np.random.randint(len(edges))
        edge = list(edges)[ind]

        reg = circuit.dag.edges[edge]['reg']
        label = edge[2]

        new_id = circuit._unique_node_id()
        new_edges = [(edge[0], new_id, label), (new_id, edge[1], label)]

        op = np.random.choice(self.single_qubit_ops, p=e_dist)
        gate = SingleQubitGateWrapper(op, reg_type='e', register=reg)
        circuit._open_qasm_update(gate)

        circuit.dag.add_node(new_id, op=gate)
        circuit.dag.remove_edges_from([edge])  # remove the edge

        circuit.dag.add_edges_from(new_edges, reg_type='e', reg=reg)
        return

    def add_emitter_CNOT(self, circuit):
        """
        Randomly selects two valid edges on which to add a new two-qubit gate.
        One edge is selected from all edges, and then the second is selected that maintains proper temporal ordering
        of the operations.
        """
        edges = [edge for edge in circuit.dag.edges if circuit.dag.edges[edge]['reg_type'] == 'e' and type(circuit.dag.nodes[edge[1]]['op']) is not Output]

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
        gate = CNOT(control=circuit.dag.edges[edge0]['reg'], control_type='e',
                                           target=circuit.dag.edges[edge1]['reg'], target_type='e')
        circuit._open_qasm_update(gate)
        circuit.dag.add_node(new_id, op=gate)

        for edge in [edge0, edge1]:
            reg = circuit.dag.edges[edge]['reg']
            label = edge[2]
            new_edges = [(edge[0], new_id, label), (new_id, edge[1], label)]

            circuit.dag.add_edges_from(new_edges, reg_type='e', reg=reg)

        circuit.dag.remove_edges_from([edge0, edge1])  # remove the selected edges
        return

    def remove_op(self, circuit, fixed_node, node=None):
        """
        Randomly selects
        """
        if node is None:
            nodes = [node for node in circuit.dag.nodes if not (
                        not (type(circuit.dag.nodes[node]['op']) not in self.fixed_ops) or not (
                            node not in fixed_node))]
            if len(nodes) == 0:
                warnings.warn("No nodes that can be removed in the circuit. Skipping.")
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
        return


def get_emission_assignment(n_photon, n_emitter, seed):
    if n_emitter == 1:
        return n_photon * [0]
    else:
        np.random.seed(seed)
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


def get_measurement_assignment(n_photon, n_emitter, seed):
    np.random.seed(seed)
    assignment = []

    for i in range(n_emitter):
        ind = np.random.randint(n_photon)
        assignment.append(ind)

    return assignment
