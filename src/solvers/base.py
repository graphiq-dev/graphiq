"""

"""
from abc import ABC, abstractmethod
import numpy as np

from src.metrics import MetricBase
from src.circuit import CircuitDAG, CircuitBase
from src.backends.compiler_base import CompilerBase


class SolverBase(ABC):
    """
    Base class for solvers, which each define an algorithm for identifying circuits towards the generation of the
    target quantum state.
    """
    def __init__(self, target, metric: MetricBase, circuit: CircuitBase, compiler: CompilerBase, *args, **kwargs):
        self.name = "base"
        self.target = target
        self.circuit = circuit
        self.metric = metric
        self.compiler = compiler

    @abstractmethod
    def solve(self, *args):
        raise NotImplementedError("Base Solver class, solver method is not implemented.")
