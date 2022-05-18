

from src.solvers.base import SolverBase
from src.metrics import MetricBase
from src.circuit import CircuitDAG, CircuitBase
from src.backends.compiler import CompilerBase


class RandomSearchSolver(SolverBase):
    """
    Implements a random search solver.
    This will randomly add/delete operations in the circuit.
    """

    n_stop = 100  # maximum number of iterations

    def __init__(self, target, metric: MetricBase, compiler: CompilerBase, *args, **kwargs):
        super().__init__(target, metric, compiler, *args, **kwargs)
        self.name = "random-search"

    def solve(self, circuit):  # TODO: move circuit to init
        for i in range(self.n_stop):
            # TODO: evolve circuit in some defined way
            circuit = circuit

            # TODO: compile/simulate the newly evolved circuit
            state = self.compiler.compile(circuit)
            print(state)

            # TODO: evaluate the state/circuit of the newly evolved circuit
            self.metric.evaluate(state, circuit)

        return circuit
