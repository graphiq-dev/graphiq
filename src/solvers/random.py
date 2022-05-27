

from src.solvers.base import SolverBase
from src.metrics import MetricBase
from src.circuit import CircuitBase
from src.backends.compiler_base import CompilerBase


class RandomSearchSolver(SolverBase):
    """
    Implements a random search solver.
    This will randomly add/delete operations in the circuit.
    """

    n_stop = 100  # maximum number of iterations

    def __init__(self, target, metric: MetricBase, circuit: CircuitBase, compiler: CompilerBase, *args, **kwargs):
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
