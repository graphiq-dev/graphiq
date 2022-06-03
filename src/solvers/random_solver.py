import matplotlib.pyplot as plt

import numpy as np
import networkx as nx
import warnings

from src.solvers.base import SolverBase
from src.metrics import MetricBase
from src.circuit import CircuitBase
from src.backends.compiler_base import CompilerBase

from src.circuit import CircuitDAG
from src import ops

from src.backends.density_matrix.compiler import DensityMatrixCompiler

from benchmarks.circuits import *


class RandomSearchSolver(SolverBase):
    """
    Implements a random search solver.
    This will randomly add/delete operations in the circuit.
    """

    n_stop = 40  # maximum number of iterations
    fixed_ops = [  # ops that should never be removed/swapped
        ops.Input,
        ops.Output
    ]

    def __init__(self,
                 target=None,
                 metric=None,
                 circuit=None,
                 compiler=None,
                 *args, **kwargs):
        super().__init__(target, metric, circuit, compiler, *args, **kwargs)
        self.name = "random-search"

        self.transformations = [
            self.add_one_qubit_op,
            # self.add_two_qubit_op,
            # self.remove_op
        ]

    def solve(self):
        circuit = self.circuit

        for i in range(self.n_stop):
            # TODO: evolve circuit in some defined way
            print(f"\nNew generation {i}")
            transformation = self.transformations[np.random.randint(len(self.transformations))]
            print(transformation)

            transformation(circuit)
            circuit.validate()
            # circuit.draw_dag()
            plt.pause(0.05)
            # TODO: compile/simulate the newly evolved circuit
            # state = self.compiler.compile(self.circuit)
            # print(state)
            state = self.compiler.compile(circuit)
            # TODO: evaluate the state/circuit of the newly evolved circuit
            val = self.metric.evaluate(state, circuit)
            print(val)
        return

    def add_one_qubit_op(self, circuit):
        """
        Randomly selects one valid edge on which to add a new single-qubit gate
        """
        edges = list(circuit.dag.edges)

        ind = np.random.randint(len(edges))
        edge = list(edges)[ind]

        reg = circuit.dag.edges[edge]['reg']
        bit = circuit.dag.edges[edge]['bit']

        new_id = circuit._unique_node_id()
        new_edges = [(edge[0], new_id), (new_id, edge[1])]

        circuit.dag.add_node(new_id, op=ops.Hadamard(register=reg))
        circuit.dag.remove_edges_from([edge])  # remove the edge

        circuit.dag.add_edges_from(new_edges, reg_type='q', reg=reg, bit=bit)

        return circuit

    def add_two_qubit_op(self, circuit):
        """
        Randomly selects two valid edges on which to add a new two-qubit gate.
        One edge is selected from all edges, and then the second is selected that maintains proper temporal ordering
        of the operations.
        """
        edges = [edge for edge in circuit.dag.edges if circuit.dag.edges[edge]['reg_type'] == 'q']

        ind = np.random.randint(len(edges))
        edge0 = list(edges)[ind]

        ancestors = nx.ancestors(circuit.dag, edge0[0])
        descendants = nx.descendants(circuit.dag, edge0[1])

        ancestor_edges = list(circuit.dag.in_edges(edge0[0]))
        for anc in ancestors:
            ancestor_edges.extend(nx.edges(circuit.dag, anc))

        descendant_edges = list(circuit.dag.out_edges(edge0[1]))
        for des in descendants:
            descendant_edges.extend(nx.edges(circuit.dag, des))

        possible_edges = set(edges) - set([edge0]) - set(ancestor_edges) - set(descendant_edges)
        if len(possible_edges) == 0:
            warnings.warn("No valid registers to place the two-qubit gate")
            return

        ind = np.random.randint(len(possible_edges))
        edge1 = list(possible_edges)[ind]

        new_id = circuit._unique_node_id()

        # print(edge0, edge1)
        print("Add 2-qubit", new_id, edge0, circuit.dag.edges[edge0]['reg'], edge1, circuit.dag.edges[edge1]['reg'])
        assert edge0 != edge1
        assert edge0 is not None
        # TODO: remove/reconnect the edges while adding a new two-qubit gate

        circuit.dag.add_node(new_id, op=ops.CNOT(control=circuit.dag.edges[edge0]['reg'],
                                                 target=circuit.dag.edges[edge1]['reg']))

        for edge in [edge0, edge1]:
            reg = circuit.dag.edges[edge]['reg']
            bit = circuit.dag.edges[edge]['bit']
            new_edges = [(edge[0], new_id, reg), (new_id, edge[1], reg)]

            circuit.dag.add_edges_from(new_edges, reg_type='q', reg=reg, bit=bit)

        circuit.dag.remove_edges_from([edge0, edge1])  # remove the selected edges
        return circuit

    def remove_op(self, circuit, node=None):
        """
        Randomly selects
        """

        if node is None:
            nodes = [node for node in circuit.dag.nodes if type(circuit.dag.nodes[node]['op']) not in self.fixed_ops]
            if len(nodes) == 0:
                warnings.warn("No nodes that can be removed in the circuit. Skipping.")
                return

            ind = np.random.randint(len(nodes))
            node = nodes[ind]

        in_edges = list(circuit.dag.in_edges(node))
        out_edges = list(circuit.dag.out_edges(node))
        print("Remove", node, circuit.dag.nodes[node]['op'], in_edges, out_edges)
        print(f"\t{[circuit.dag.edges[e] for e in in_edges]}")
        print(f"\t{[circuit.dag.edges[e] for e in out_edges]}")
        for in_edge in in_edges:
            reg = circuit.dag.edges[in_edge]['reg']
            for out_edge in out_edges:
                if circuit.dag.edges[in_edge]['reg'] == circuit.dag.edges[out_edge]['reg']:
                    circuit.dag.add_edge(in_edge[0], out_edge[1], reg_type='q', reg=reg, bit=0)
                    print(f"We are connecting the in/out edges.")
        circuit.dag.remove_node(node)
        return


if __name__ == "__main__":
    # circuit, _ = bell_state_circuit()
    # circuit, _ = ghz3_state_circuit()
    # circuit, _ = ghz4_state_circuit()
    np.random.seed(3)
    plt.ion()
    circuit = CircuitDAG(n_quantum=15, n_classical=0)

    # circuit.draw_dag()
    solver = RandomSearchSolver()

    def check_cnots(cir: CircuitDAG):
        for node in cir.dag.nodes:
            if type(cir.dag.nodes[node]['op']) is ops.CNOT:
                assert len(circuit.dag.in_edges(node)) == 2, f"in edges is {circuit.dag.in_edges(node)} not 2"
                assert len(circuit.dag.out_edges(node)) == 2, f"out edges is {circuit.dag.out_edges(node)} not 2"

    # for node in [10, 11, 9]:
    for i in range(1):
        n = 200
        # solver.add_one_qubit_op(circuit)
        for j in range(n):
            # solver.add_one_qubit_op(circuit)
            check_cnots(circuit)
            circuit = solver.add_two_qubit_op(circuit)
            check_cnots(circuit)
            circuit.validate()
            plt.pause(0.01)
            print(f"{j}")
        # for j in range(n):
        #     solver.remove_op(circuit)
        #     check_cnots(circuit)
        #     circuit.validate()
        #     print(f"{j}")
        #     plt.pause(0.001)

        print(i)


        # circuit.draw_dag()
        # plt.pause(0.001)
        # try:
        #     circuit.validate()
        # except:
        #     circuit.draw_dag()
        #     plt.pause(0.001)
        #     plt.show()
        #     break

        # if i % 1 == 0:
        #     fig, axs = circuit.draw_dag()

        plt.show()
        # fig.suptitle(f"{i}")
        # plt.show()
    # input()