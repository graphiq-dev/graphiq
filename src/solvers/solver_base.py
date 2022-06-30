"""
Solvers are implementations of search algorithms to find quantum circuits which produce a target state.

Solvers have a main function, .solve(), which runs the search algorithm and return a Hall of Fame circuits.
"""

from abc import ABC, abstractmethod

from src.metrics import MetricBase
from src.circuit import CircuitBase
from src.backends.compiler_base import CompilerBase
from src import ops

import numpy as np
import random
import warnings
import copy


class SolverBase(ABC):
    """
    Base class for solvers, which each define an algorithm for identifying circuits towards the generation of the
    target quantum state.
    """

    def __init__(
        self,
        target,
        metric: MetricBase,
        compiler: CompilerBase,
        circuit: CircuitBase = None,
    ):
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
        raise NotImplementedError(
            "Base Solver class, solver method is not implemented."
        )

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


class RandomSearchSolver(SolverBase):
    """
    Implements a random search solver.
    This will randomly add/delete operations in the circuit.
    """

    name = "random-search"
    n_stop = 10  # maximum number of iterations
    n_hof = 10
    n_pop = 10

    fixed_ops = [ops.Input, ops.Output]  # ops that should never be removed/swapped

    single_qubit_ops = [
        ops.Hadamard,
    ]

    two_qubit_ops = [
        ops.CNOT,
    ]

    def __init__(
        self,
        target,
        metric: MetricBase,
        compiler: CompilerBase,
        circuit: CircuitBase = None,
        *args,
        **kwargs,
    ):

        super().__init__(target, metric, compiler, circuit)

        # hof stores the best circuits and their scores in the form of: (scores, circuits)
        self.hof = [(np.inf, None) for _ in range(self.n_hof)]

        self.trans_probs = {None: None}

        self.transformations = list(self.trans_probs.keys())

    def update_hof(self, population):
        """
        Updates the Hall-of-Fame (HOF), which is a list of the best circuits encountered so far in the search.

        :param population: population of the circuits, a list of (score, circuit) tuples
        :type population: list[(float, CircuitBase)]
        :return: nothing
        """
        for score, circuit in population:
            for i in range(self.n_hof):
                if np.isclose(score, self.hof[i][0]):
                    if len(circuit.dag.nodes) < len(self.hof[i][1].dag.nodes):
                        self.hof.insert(i, (score, copy.deepcopy(circuit)))
                        self.hof.pop()
                        break

                elif score < self.hof[i][0]:
                    self.hof.insert(i, (score, copy.deepcopy(circuit)))
                    self.hof.pop()
                    break

    def tournament_selection(self, population, k=2):
        """
        Tournament selection for choosing the next generation population to be mutated/crossed-over

        :param population: population of the circuits, a list of (score, circuit) tuples
        :type population: list[(float, CircuitBase)]
        :param k: size of the tournament
        :type k: int
        :return: a new population
        :rtype: list[(float, CircuitBase)]
        """
        if k == 0:  # in this case, no selection
            return population

        population_new = []
        for i in range(self.n_pop):
            # select the sub-population for the tournament with uniform random
            tourn_pop = random.choices(population, k=k)

            # sort the tournament population by the score, such that the first index is the lowest scoring circuit
            # tourn_pop.sort(key=lambda x: x[0], reverse=False)
            best = min(tourn_pop, key=lambda x: x[0])

            population_new.append(
                copy.deepcopy(best)
            )  # add the best performing circuit in the tournament

        return population_new

    def solve(self, *args):
        raise NotImplementedError(
            "Base Solver class, solver method is not implemented."
        )
