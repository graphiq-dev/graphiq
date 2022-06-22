"""
Solvers are implementations of search algorithms to find quantum circuits which produce a target state.

Solvers have a main function, .solve(), which runs the search algorithm and return a Hall of Fame circuits.
"""

from abc import ABC, abstractmethod

from src.metrics import MetricBase
from src.circuit import CircuitBase
from src.backends.compiler_base import CompilerBase

import numpy as np
import random
import warnings


class SolverBase(ABC):
    """
    Base class for solvers, which each define an algorithm for identifying circuits towards the generation of the
    target quantum state.
    """
    def __init__(self, target, metric: MetricBase, compiler: CompilerBase, circuit: CircuitBase = None):
        self.name = "base"
        self.target = target
        self.metric = metric
        self.compiler = compiler

        if circuit is None:
            warnings.warn(f"Initial circuit for {self.__class__.__name__} is 'None'. ")
        self.circuit = circuit

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
