"""
Solvers are implementations of search algorithms to find quantum circuits which produce a target state.

Solvers have a main function, .solve(), which runs the search algorithm and return a Hall of Fame circuits.
"""

from abc import ABC, abstractmethod
import numpy as np
import random
import warnings
import copy

from src.metrics import MetricBase
from src.circuit import CircuitBase
from src.backends.compiler_base import CompilerBase
from src import ops
from src.io import IO
import src.noise.noise_models as nm


class SolverBase(ABC):
    """
    Base class for solvers, which each define an algorithm for identifying circuits towards the generation of the
    target quantum state.
    """

    name = "base"

    fixed_ops = [ops.Input, ops.Output]  # ops that should never be removed/swapped

    one_qubit_ops = [
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
        io: IO = None,
    ):
        self.target = target
        self.metric = metric
        self.compiler = compiler
        self.io = io

        self.circuit = circuit

        self.last_seed = None
        self.logs = {}
        self.result = None

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

    def _wrap_noise(self, op, noise_model_mapping):
        """
        A helper function to consolidate noise models for OneQubitGateWrapper operation

        :param op: a list of operations
        :type op: list[ops.OperationBase]
        :param noise_model_mapping: a dictionary that stores the mapping between an operation
            and its associated noise model
        :type noise_model_mapping: dict
        :return: a list of noise models
        :rtype: list[nm.NoiseBase]
        """

        if "OneQubitGateWrapper" in noise_model_mapping:
            noise = noise_model_mapping["OneQubitGateWrapper"]
        else:
            noise = []
            for each_op in op:
                noise.append(
                    self._identify_noise(each_op.__name__, noise_model_mapping)
                )
        return noise

    def _identify_noise(self, op, noise_model_mapping):
        """
        A helper function to identify the noise model for an operation

        :param op: an operation or its name
        :type op: ops.OperationBase or str
        :param noise_model_mapping: a dictionary that stores the mapping between an operation
            and its associated noise model
        :type noise_model_mapping: dict
        :return: a noise model
        :rtype: nm.NoiseBase
        """
        if type(op) != str:
            op_name = type(op).__name__
        else:
            op_name = op
        if op_name in noise_model_mapping.keys():
            return noise_model_mapping[op_name]
        else:
            return nm.NoNoise()


class RandomSolverSetting(ABC):
    def __init__(
        self,
        n_hof=5,
        n_stop=50,
        n_pop=50,
    ):
        self._n_hof = n_hof
        self._n_stop = n_stop
        self._n_pop = n_pop

    @property
    def n_hof(self):
        return self._n_hof

    @n_hof.setter
    def n_hof(self, value):
        assert type(value) == int
        self._n_hof = value

    @property
    def n_stop(self):
        return self._n_stop

    @n_stop.setter
    def n_stop(self, value):
        assert type(value) == int
        self._n_stop = value

    @property
    def n_pop(self):
        return self._n_hof

    @n_pop.setter
    def n_pop(self, value):
        assert type(value) == int
        self._n_pop = value

    def __str__(self):
        s = f"n_hof = {self.n_hof}\n"
        s += f"n_pop = {self.n_pop}\n"
        s += f"n_stop = {self.n_stop}\n"
        return s


class RandomSearchSolver(SolverBase):
    """
    Implements a random search solver.
    This will randomly add/delete operations in the circuit.
    """

    name = "random-search"

    def __init__(
        self,
        target,
        metric: MetricBase,
        compiler: CompilerBase,
        circuit: CircuitBase = None,
        io: IO = None,
        solver_setting=RandomSolverSetting(),
        *args,
        **kwargs,
    ):

        super().__init__(target, metric, compiler, circuit, io)

        # hof stores the best circuits and their scores in the form of: (scores, circuits)
        self.setting = solver_setting
        self.hof = [(np.inf, None) for _ in range(self.setting.n_hof)]
        self.trans_probs = {None: None}
        self.transformations = list(self.trans_probs.keys())
        self.logs = {"population": [], "hof": []}

    def update_hof(self, population):
        """
        Updates the Hall-of-Fame (HOF), which is a list of the best circuits encountered so far in the search.

        :param population: population of the circuits, a list of (score, circuit) tuples
        :type population: list[(float, CircuitBase)]
        :return: nothing
        """
        for score, circuit in population:
            for i in range(self.setting.n_hof):
                if np.isclose(score, self.hof[i][0]):
                    if len(circuit.dag.nodes) < len(self.hof[i][1].dag.nodes):
                        self.hof.insert(i, (score, circuit.copy()))
                        self.hof.pop()
                        break

                elif score < self.hof[i][0]:
                    self.hof.insert(i, (score, circuit.copy()))
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
        for i in range(self.setting.n_pop):
            # select the sub-population for the tournament with uniform random
            tourn_pop = random.choices(population, k=k)

            # sort the tournament population by the score, such that the first index is the lowest scoring circuit
            # tourn_pop.sort(key=lambda x: x[0], reverse=False)
            best = min(tourn_pop, key=lambda x: x[0])

            # add the best performing circuit in the tournament
            population_new.append(copy.deepcopy(best))
        return population_new

    def solve(self, *args):
        raise NotImplementedError(
            "Base Solver class, solver method is not implemented."
        )
