"""

"""
from abc import ABC, abstractmethod

from src.metrics import MetricBase
from src.circuit import CircuitBase
from src.backends.compiler_base import CompilerBase

import numpy as np
import random


class SolverBase(ABC):
    """
    Base class for solvers, which each define an algorithm for identifying circuits towards the generation of the
    target quantum state.
    """
    def __init__(self, target, metric: MetricBase, circuit: CircuitBase, compiler: CompilerBase):
        self.name = "base"
        self.target = target
        self.circuit = circuit
        self.metric = metric
        self.compiler = compiler
        self.last_seed = None

    @abstractmethod
    def solve(self, *args):
        raise NotImplementedError("Base Solver class, solver method is not implemented.")

    @staticmethod
    def seed(seed=None):
        """
        Sets the seed for both the numpy.random and random packages.
        Accessible by any Solver subclass

        :param seed: a random number seed
        :type seed: int or None
        :return: nothing
        """
        np.random.seed(seed)
        random.seed(seed)
        SolverBase.last_seed = seed

    @property
    def solver_info(self):
        return {}
