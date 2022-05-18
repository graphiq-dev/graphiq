"""
Solver implementations, which are different algorithms for searching for high-performing circuits.
"""

from abc import ABC, abstractmethod
import numpy as np

from src.metrics import MetricBase
from src.circuit import CircuitDAG, CircuitBase
from src.backends.compiler import CompilerBase


class SolverBase(ABC):
    """
    Base class for solvers, which each define an algorithm for identifying circuits towards the generation of the
    target quantum state.
    """
    def __init__(self, target, metric: MetricBase, compiler: CompilerBase, *args, **kwargs):
        self.name = "base"
        self.target = target

        self.metric = metric
        self.compiler = compiler

    @abstractmethod
    def solve(self, *args):
        raise NotImplementedError("Base Solver class, solver method is not implemented.")


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


