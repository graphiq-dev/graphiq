
import numpy as np
import networkx as nx

from src.solvers.base import SolverBase
from src.metrics import MetricBase
from src.circuit import CircuitBase
from src.backends.compiler_base import CompilerBase

from src.circuit import CircuitDAG
from src import ops

from src.backends.density_matrix.compiler import DensityMatrixCompiler

from src.libraries.circuits import *


class RandomSearchSolver(SolverBase):
    """
    Implements a random search solver.
    This will randomly add/delete operations in the circuit.
    """

    n_stop = 100  # maximum number of iterations
    fixed_ops = [ops.Input, ops.Output]  # ops that should never be removed/swapped

    def __init__(self,
                 target=None,
                 metric=None,
                 circuit=None,
                 compiler=None,
                 *args, **kwargs):
        super().__init__(target, metric, circuit, compiler, *args, **kwargs)
        self.name = "random-search"

    def solve(self):
        for i in range(self.n_stop):
            # TODO: evolve circuit in some defined way

            # TODO: compile/simulate the newly evolved circuit
            state = self.compiler.compile(self.circuit)
            print(state)

            # TODO: evaluate the state/circuit of the newly evolved circuit
            self.metric.evaluate(state, self.circuit)

        return

    def add_one_qubit(self, circuit):
        """
        Randomly selects one valid edge on which to add a new single-qubit gate
        """
        edges = list(circuit.dag.edges)

        ind = np.random.randint(len(edges))
        edge = list(edges)[ind]
        print(edge)
        return

    def add_two_qubit(self, circuit):
        """
        Randomly selects two valid edges on which to add a new two-qubit gate.
        One edge is selected from all edges, and then the second is selected that maintains proper temporal ordering
        of the operations.
        """
        edges = list(circuit.dag.edges)  # TODO: this should only be quantum register edges

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

        ind = np.random.randint(len(possible_edges))
        edge1 = list(possible_edges)[ind]

        print(edge0, edge1)

        # TODO: remove/reconnect the edges while adding a new two-qubit gate
        # TODO: not all of the edges have the same data fields, i.e. many do not have the 'reg' or others.
        # this should be consistent
        return

    def remove_one_qubit(self, circuit):
        """
        Randomly selects
        """
        nodes = [node for node in circuit.dag.nodes if type(circuit.dag.nodes[node]['op']) not in self.fixed_ops]

        ind = np.random.randint(len(nodes))
        node = nodes[ind]
        print(nodes, node)
        return


if __name__ == "__main__":
    # circuit, _ = bell_state_circuit()
    circuit, _ = ghz4_state_circuit()
    circuit.show()

    solver = RandomSearchSolver()
    solver.add_one_qubit(circuit)
    solver.add_two_qubit(circuit)
    solver.remove_one_qubit(circuit)
