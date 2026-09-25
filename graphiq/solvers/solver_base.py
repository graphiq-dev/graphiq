# Copyright (c) 2022-2024 Quantum Bridge Technologies Inc.
# Copyright (c) 2022-2024 Ki3 Photonics Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Solvers are implementations of search algorithms to find quantum circuits which produce a target state.

Solvers have a main function, .solve(), which runs the search algorithm and return a Hall of Fame circuits.
"""

import copy
import random
from abc import ABC, abstractmethod

import numpy as np

import graphiq.noise.noise_models as nm
from graphiq.backends.compiler_base import CompilerBase
from graphiq.circuit.circuit_base import CircuitBase, ops
from graphiq.io import IO
from graphiq.metrics import MetricBase


class SolverBase(ABC):
    """
    Base class for solvers, which each define an algorithm for identifying circuits towards the generation of the
    target quantum state.
    """

    name = "base"

    fixed_ops = [ops.Input, ops.Output]  # ops that should never be removed/swapped

    one_qubit_ops = [ops.Hadamard, ops.Phase, ops.SigmaX, ops.SigmaY, ops.SigmaZ]

    two_qubit_ops = [
        ops.CNOT,
    ]

    # the last seed passed to seed(); a class attribute because seed() sets the
    # process-global numpy/random seeds and records it on SolverBase itself
    last_seed = None

    def __init__(
        self,
        target,
        metric: MetricBase,
        compiler: CompilerBase,
        circuit: CircuitBase = None,
        io: IO = None,
        solver_setting=None,
    ):
        """
        Create a SolverBase object

        :param target: the target state the solver searches for a circuit to generate
        :type target: QuantumState
        :param metric: the objective the solver minimizes
        :type metric: MetricBase
        :param compiler: the compiler used to simulate each candidate circuit
        :type compiler: CompilerBase
        :param circuit: an initial circuit to start from, or None
        :type circuit: CircuitBase or None
        :param io: an IO object used to save results, or None for no saving
        :type io: IO or None
        :param solver_setting: the solver's own settings object, or None
        :type solver_setting: object or None
        :return: nothing
        :rtype: None
        """
        self.target = target
        self.metric = metric
        self.compiler = compiler
        self.io = io

        self.circuit = circuit
        self.solver_setting = solver_setting
        self.logs = {}
        self.result = None

    @abstractmethod
    def solve(self, *args):
        """
        Run the search. Each concrete solver implements its own.

        :raises NotImplementedError: always, on this base class
        :return: nothing
        :rtype: None
        """
        raise NotImplementedError(
            "Base Solver class, solver method is not implemented."
        )

    @staticmethod
    def seed(seed=None):
        """
        Sets the seed for both the numpy.random and random packages.
        Accessible by any Solver subclass

        The seed is recorded in last_seed, a class-level attribute on SolverBase (reported as
        solver_info["seed"] by the solvers that log it). Like the numpy.random and random seeds it
        sets, it is shared by the whole process: a solver reports the seed most recently set by any
        solver, not necessarily its own. Measurement outcomes in the stabilizer and density matrix
        backends are drawn from the global numpy.random state, so this seed does govern them.
        It does not reach components that build their own unseeded np.random.default_rng()
        (for example, AlternateTargetSolver's Monte Carlo seeding when a seed is missing, and the
        random graph generators in benchmarks/graph_states.py when no np_rng is given).

        :param seed: a random number seed
        :type seed: int or None
        :return: nothing
        """
        np.random.seed(seed)
        random.seed(seed)
        SolverBase.last_seed = seed

    @property
    def solver_info(self):
        """
        Solver-specific information to record alongside a result.

        Empty on the base class; a subclass overrides it with whatever describes its own search.

        :return: a dictionary of solver information
        :rtype: dict
        """
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
                noise.append(self._identify_noise(each_op, noise_model_mapping))
        return noise

    def _identify_noise(self, op, noise_model_mapping):
        """
        A helper function to identify the noise model for an operation

        :param op: an operation
        :type op: ops.OperationBase
        :param noise_model_mapping: a dictionary that stores the mapping between an operation
            and its associated noise model. A two-legged op is looked up by its per-leg keys,
            ``<Op>_control`` and ``<Op>_target``, first; the bare ``<Op>`` key is only used as a
            fallback when neither per-leg key is present, in which case the same model is
            returned for both legs
        :type noise_model_mapping: dict
        :return: a noise model
        :rtype: nm.NoiseBase
        """
        # first make sure whether the passed operations are instances (of a class) or classes.
        if isinstance(op, ops.OperationBase):
            # if op is an instance, turn it into the class itself.
            op = type(op)
        # op is a class here, so this must be issubclass, not isinstance
        if issubclass(
            op,
            (ops.ControlledPairOperationBase, ops.ClassicalControlledPairOperationBase),
        ):
            op_name = op.__name__
            op_control = op.__name__ + "_control"
            op_target = op.__name__ + "_target"
            if op_control in noise_model_mapping.keys():
                control_noise = noise_model_mapping[op_control]
            else:
                control_noise = nm.NoNoise()
            if op_target in noise_model_mapping.keys():
                target_noise = noise_model_mapping[op_target]
            else:
                target_noise = nm.NoNoise()
            if op_name in noise_model_mapping.keys() and type(
                target_noise
            ) == nm.NoNoise == type(control_noise):
                return [noise_model_mapping[op_name], noise_model_mapping[op_name]]
            else:
                return [control_noise, target_noise]
        else:
            op_name = op.__name__
            if op_name in noise_model_mapping.keys():
                return noise_model_mapping[op_name]
            else:
                return nm.NoNoise()


class RandomSearchSolverSetting(ABC):
    """
    The settings every random-search-style solver shares: hall-of-fame size, how long to run,
    and how many candidates to hold per iteration.
    """

    def __init__(
        self,
        n_hof=5,
        n_stop=50,
        n_pop=50,
    ):
        """
        Create a RandomSearchSolverSetting object

        :param n_hof: the number of best circuits kept in the hall of fame
        :type n_hof: int
        :param n_stop: the number of iterations to run before stopping
        :type n_stop: int
        :param n_pop: the number of candidate circuits held per iteration
        :type n_pop: int
        :return: nothing
        :rtype: None
        """
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
        return self._n_pop

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
        solver_setting=None,
        *args,
        **kwargs,
    ):
        """
        Create a RandomSearchSolver object

        :param target: the target state the solver searches for a circuit to generate
        :type target: QuantumState
        :param metric: the objective the solver minimizes
        :type metric: MetricBase
        :param compiler: the compiler used to simulate each candidate circuit
        :type compiler: CompilerBase
        :param circuit: an initial circuit to start from, or None
        :type circuit: CircuitBase or None
        :param io: an IO object used to save results, or None for no saving
        :type io: IO or None
        :param solver_setting: the hall-of-fame size, stopping point and population size, stored as
            self.setting; a fresh default is built when None
        :type solver_setting: RandomSearchSolverSetting or None
        :return: nothing
        :rtype: None
        """
        super().__init__(target, metric, compiler, circuit, io)

        # built per instance: a default instance in the signature would be
        # shared (and mutated) by every solver constructed without one
        if solver_setting is None:
            solver_setting = RandomSearchSolverSetting()

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
                    # self.hof[i][1] is None for a still-unfilled placeholder
                    # slot (initialized to (np.inf, None)); any real circuit
                    # is preferred over an empty slot.
                    if self.hof[i][1] is None or len(circuit.dag.nodes) < len(
                        self.hof[i][1].dag.nodes
                    ):
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
        """
        Run the search. Not implemented on RandomSearchSolver; a subclass such as
        EvolutionarySolver overrides it.

        :raises NotImplementedError: always, on this class
        :return: nothing
        :rtype: None
        """
        raise NotImplementedError(
            "Base Solver class, solver method is not implemented."
        )
