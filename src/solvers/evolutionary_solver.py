import matplotlib.pyplot as plt

import numpy as np
import networkx as nx
import warnings
import collections
import copy
import time
import random

from src.solvers.base import SolverBase
from src.metrics import MetricBase
from src.circuit import CircuitBase
from src.backends.compiler_base import CompilerBase

from src.backends.density_matrix.compiler import DensityMatrixCompiler
from src.circuit import CircuitDAG
from src.metrics import MetricFidelity

from src.visualizers.density_matrix import density_matrix_bars
from src import ops
from src.io import IO

from src.solvers.random_solver import RandomSearchSolver
from benchmarks.circuits import *


class EvolutionarySolver(RandomSearchSolver):
    """
    Implements an evolutionary search solver.
    This will randomly add/delete operations in the circuit.
    """

    name = "evolutionary-search"
    n_stop = 40  # maximum number of iterations
    n_hof = 10
    n_pop = 10

    tournament_k = 2  # tournament size for selection of the next population

    fixed_ops = [  # ops that should never be removed/swapped
        ops.Input,
        ops.Output
    ]

    single_qubit_ops = [
        ops.Hadamard,
        ops.SigmaX,
        ops.SigmaY,
        ops.SigmaZ,
    ]

    two_qubit_ops = [
        ops.CNOT,
        ops.CPhase,
    ]

    def __init__(self,
                 target=None,
                 metric=None,
                 circuit=None,
                 compiler=None,
                 *args, **kwargs):
        super().__init__(target, metric, circuit, compiler, *args, **kwargs)

        self.transformations = [
            self.add_one_qubit_op,
            self.add_two_qubit_op,
            self.remove_op
        ]

        self.trans_probs = {
            self.add_two_qubit_op: 0.5,
            self.add_one_qubit_op: 0.5,
            self.remove_op: 0.0
        }

    def solve(self):
        """

        :return:
        """

        population = [(None, copy.deepcopy(self.circuit)) for _ in range(self.n_pop)]

        for i in range(self.n_stop):
            for j in range(self.n_pop):

                transformation = np.random.choice(list(self.trans_probs.keys()), p=list(self.trans_probs.values()))

                circuit = population[j][1]
                transformation(circuit)
                # print(f"{transformation.__name__}")

                circuit.validate()

                state = self.compiler.compile(circuit)  # this will pass out a density matrix object
                score = self.metric.evaluate(state.data, circuit)

                population[j] = (score, circuit)

            self.update_hof(population)
            self.adapt_probabilities(i)
            population = self.tournament_selection(population, self.tournament_k)

            print(f"New generation {i} | {self.hof[0][0]:.4f}")

        return

    def adapt_probabilities(self, iteration: int):
        self.trans_probs[self.add_one_qubit_op] = (1.0 - iteration / self.n_stop) / 2
        self.trans_probs[self.add_two_qubit_op] = (1.0 - iteration / self.n_stop) / 2
        self.trans_probs[self.remove_op] = iteration / self.n_stop
        return

    def tournament_selection(self, population, k=2):
        """
        Tournament selection for choosing the next generation population to be mutated/crossed-over


        :param population: population of the circuits, a list of (score, circuit) tuples
        :param k: size of the tournament
        :return:
        """
        population_new = []
        for i in range(self.n_pop):
            # select the sub-population for the tournament with uniform random
            tourn_pop = random.choices(population, k=k)

            # sort the tournament population by the score, such that the first index is the lowest scoring circuit
            tourn_pop.sort(key=lambda x: x[0], reverse=False)

            population_new.append(copy.deepcopy(tourn_pop[0]))  # add the best performing circuit in the tournament

        return population_new


def sort_hof(hof):
    hof0_score = hof[0][0]
    hof0 = hof[0][1]
    for (score, cir) in hof:
        print(score, len(cir.dag.nodes))
        if np.isclose(score, hof0_score):
            if len(cir.dag.nodes) < len(hof0.dag.nodes):
                print("smaller circuit")
                hof0 = cir
    return hof0


if __name__ == "__main__":
    #%% here we have access
    EvolutionarySolver.n_stop = 40
    EvolutionarySolver.n_pop = 150
    EvolutionarySolver.n_hof = 10
    EvolutionarySolver.tournament_k = 10

    #%% comment/uncomment for reproducibility
    # seed = 0
    # np.random.seed(seed)
    # random.seed(seed)
    # %% select which state we want to target
    # circuit_ideal, state_ideal = bell_state_circuit()
    # circuit_ideal, state_ideal = ghz3_state_circuit()
    # circuit_ideal, state_ideal = linear_cluster_3qubit_circuit()
    # circuit_ideal, state_ideal = ghz4_state_circuit()
    circuit_ideal, state_ideal = linear_cluster_4qubit_circuit()

    #%% construct all of our important objects
    target = state_ideal['dm']
    circuit = CircuitDAG(n_quantum=4, n_classical=0)
    compiler = DensityMatrixCompiler()
    metric = MetricFidelity(target=target)

    solver = EvolutionarySolver(target=target, metric=metric, circuit=circuit, compiler=compiler)

    # circuits = [copy.deepcopy(circuit) for _ in range(10)]
    # print(solver.tournament_probs)
    # solver.tournament_selection(circuits)

    #%% call the solver.solve() function to implement the random search algorithm
    t0 = time.time()
    solver.solve()
    t1 = time.time()

    #%% print/plot the results
    print(solver.hof)
    print(f"Total time {t1-t0}")

    circuit = sort_hof(solver.hof)  # get the best hof circuit

    # extract the best performing circuit
    fig, axs = density_matrix_bars(target)
    fig.suptitle("TARGET DENSITY MATRIX")
    plt.show()

    state = compiler.compile(circuit)
    fig, axs = density_matrix_bars(state.data)
    fig.suptitle("CREATED DENSITY MATRIX")
    plt.show()

    circuit.draw_circuit()
