import matplotlib.pyplot as plt

import numpy as np
import networkx as nx
import warnings
import collections
import copy
import time

from src.solvers.base import SolverBase
from src.metrics import MetricBase
from src.circuit import CircuitBase
from src.backends.compiler_base import CompilerBase

from src.backends.density_matrix.compiler import DensityMatrixCompiler
from src.circuit import CircuitDAG
from src.metrics import Infidelity

from src.visualizers.density_matrix import density_matrix_bars
from src import ops
from src.io import IO

from benchmarks.circuits import *


class RandomSearchSolver(SolverBase):
    """
    Implements a random search solver.
    This will randomly add/delete operations in the circuit.
    """

    name = "random-search"
    n_stop = 40  # maximum number of iterations
    n_hof = 10
    n_pop = 10

    fixed_ops = [  # ops that should never be removed/swapped
        ops.Input,
        ops.Output
    ]

    single_qubit_ops = [
        ops.Hadamard,
        ops.SigmaX,
        ops.SigmaY,
        ops.SigmaZ,
    ]

    two_qubit_ops = [
        ops.CNOT,
        ops.CPhase,
    ]

    def __init__(self,
                 target=None,
                 metric=None,
                 circuit=None,
                 compiler=None,
                 *args, **kwargs):
        super().__init__(target, metric, circuit, compiler, *args, **kwargs)

        self.transformations = [
            self.add_one_qubit_op,
            self.add_two_qubit_op,
            self.remove_op
        ]

        # hof stores the best circuits and their scores in the form of: (scores, circuits)
        self.hof = [(np.inf, None) for _ in range(self.n_hof)]

        self.trans_probs = {
            self.add_two_qubit_op: 0.25,
            self.add_one_qubit_op: 0.25,
            self.remove_op: 0.5
        }

    def update_hof(self, population):
        for score, circuit in population:
            for i in range(self.n_hof):
                if np.isclose(score, self.hof[i][0]):
                    if len(circuit.dag.nodes) < len(self.hof[i][1].dag.nodes):
                        self.hof.insert(i, (score, copy.deepcopy(circuit)))
                        self.hof.pop()

                elif score < self.hof[i][0]:
                    self.hof.insert(i, (score, copy.deepcopy(circuit)))
                    self.hof.pop()
                    break

    def solve(self):
        """
        Main solver algorithm for identifying circuits which minimizes the metric function.

        Here, all aspects of the algorithm are fully random (with uniform probabilities).
        We start with an empty N-qubit circuit, and at each iteration add

        :return:
        """
        scores = [None for _ in range(self.n_pop)]
        circuits = [copy.deepcopy(self.circuit) for _ in range(self.n_pop)]

        for i in range(self.n_stop):
            for j in range(self.n_pop):

                # transformation = self.transformations[np.random.randint(len(self.transformations))]
                transformation = np.random.choice(list(self.trans_probs.keys()), p=list(self.trans_probs.values()))

                circuit = circuits[j]
                transformation(circuit)

                circuit.validate()

                state = self.compiler.compile(circuit)  # this will pass out a density matrix object
                # print(state)

                score = self.metric.evaluate(state.data, circuit)

                scores[j] = score
                circuits[j] = circuit
                print(f"New generation {i} | {score:.4f} | {transformation.__name__}")

            self.update_hof(scores, circuits)

        return

    def add_one_qubit_op(self, circuit: CircuitDAG):
        """
        Randomly selects one valid edge on which to add a new one-qubit gate.
        """

        edges = list(circuit.dag.edges)

        ind = np.random.randint(len(edges))
        edge = list(edges)[ind]

        reg = circuit.dag.edges[edge]['reg']
        label = edge[2]

        new_id = circuit._unique_node_id()
        new_edges = [(edge[0], new_id, label), (new_id, edge[1], label)]

        op = np.random.choice(self.single_qubit_ops)

        circuit.dag.add_node(new_id, op=op(register=reg))
        circuit._open_qasm_update(op)  # fixes plotting issues

        circuit.dag.remove_edges_from([edge])  # remove the edge

        circuit.dag.add_edges_from(new_edges, reg_type='q', reg=reg)
        return

    def add_two_qubit_op(self, circuit: CircuitDAG):
        """
        Randomly selects two valid edges on which to add a new two-qubit gate.
        One edge is selected from all edges, and then the second is selected that maintains proper temporal ordering
        of the operations.
        """
        # setattr(self.add_two_qubit_op, "att1", 1)

        edges = [edge for edge in circuit.dag.edges if circuit.dag.edges[edge]['reg_type'] == 'q']

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

        op = np.random.choice(self.two_qubit_ops)
        circuit.dag.add_node(new_id, op=op(control=circuit.dag.edges[edge0]['reg'],
                                           target=circuit.dag.edges[edge1]['reg']))

        circuit._open_qasm_update(op)  # fixes plotting issues

        for edge in [edge0, edge1]:
            reg = circuit.dag.edges[edge]['reg']
            label = edge[2]
            new_edges = [(edge[0], new_id, label), (new_id, edge[1], label)]

            circuit.dag.add_edges_from(new_edges, reg_type='q', reg=reg)

        circuit.dag.remove_edges_from([edge0, edge1])  # remove the selected edges
        return

    def remove_op(self, circuit: CircuitDAG):
        """
        Randomly selects a one- or two-qubit gate to remove from the circuit
        """
        nodes = [node for node in circuit.dag.nodes if type(circuit.dag.nodes[node]['op']) not in self.fixed_ops]
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
                    circuit.dag.add_edge(in_edge[0], out_edge[1], label, reg_type='q', reg=reg)

        circuit.dag.remove_node(node)
        return

    @staticmethod
    def _check_cnots(cir: CircuitDAG):
        """
        Sanity check that all CNOT gates have two in edges and two out edges
        :param cir: circuit object
        :return:
        """
        for node in cir.dag.nodes:
            if type(cir.dag.nodes[node]['op']) is ops.CNOT:
                assert len(cir.dag.in_edges(node)) == 2, f"in edges is {cir.dag.in_edges(node)} not 2"
                assert len(cir.dag.out_edges(node)) == 2, f"out edges is {cir.dag.out_edges(node)} not 2"


if __name__ == "__main__":
    #%% here we have access
    RandomSearchSolver.n_stop = 50
    RandomSearchSolver.n_pop = 300
    RandomSearchSolver.n_hof = 5

    #%% comment/uncomment for reproducibility
    # np.random.seed(190)

    # %% select which state we want to target
    # circuit_ideal, state_ideal = bell_state_circuit()
    # circuit_ideal, state_ideal = ghz3_state_circuit()
    # circuit_ideal, state_ideal = ghz4_state_circuit()
    circuit_ideal, state_ideal = linear_cluster_3qubit_circuit()
    # circuit_ideal, state_ideal = linear_cluster_4qubit_circuit()

    #%% construct all of our important objects
    target = state_ideal['dm']

    circuit = CircuitDAG(n_emitter=4, n_classical=0)
    compiler = DensityMatrixCompiler()
    metric = Infidelity(target=target)

    solver = RandomSearchSolver(target=target, metric=metric, circuit=circuit, compiler=compiler)

    #%% call the solver.solve() function to implement the random search algorithm
    t0 = time.time()
    solver.solve()
    t1 = time.time()

    #%% print/plot the results
    print(solver.hof)
    print(f"Total time {t1-t0}")

    # extract the best performing circuit
    fig, axs = density_matrix_bars(target)
    fig.suptitle("TARGET DENSITY MATRIX")
    plt.show()

    circuit = solver.hof[1][0]  # get the best hof circuit
    state = compiler.compile(circuit)
    fig, axs = density_matrix_bars(state.data)
    fig.suptitle("CREATED DENSITY MATRIX")
    plt.show()

    circuit.draw_circuit()
