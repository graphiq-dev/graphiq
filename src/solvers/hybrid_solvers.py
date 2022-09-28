"""
Contains various hybrid solvers
"""
import numpy as np

from src.solvers.evolutionary_solver import EvolutionarySolver
from src.solvers.deterministic_solver import DeterministicSolver
from src.backends.compiler_base import CompilerBase
from src.circuit import CircuitDAG
from src.metrics import MetricBase
from src.io import IO


class HybridEvolutionarySolver(EvolutionarySolver):
    """
    Implements a hybrid solver based on deterministic solver and rule-based evolutionary search solver.
    It takes the solution from DeterministicSolver (without noise simulation)
    as the starting point for the EvolutionarySolver.
    """

    name = "hybrid evolutionary-search"

    def __init__(
        self,
        target,
        metric: MetricBase,
        compiler: CompilerBase,
        io: IO = None,
        n_hof=5,
        n_stop=50,
        n_pop=50,
        tournament_k=2,
        selection_active=False,
        save_openqasm: str = "none",
        noise_model_mapping=None,
        *args,
        **kwargs,
    ):
        """
        Initialize a hybrid solver based on DeterministicSolver and EvolutionarySolver

        :param target: target quantum state
        :type target: QuantumState
        :param metric: metric (cost) function to minimize
        :type metric: MetricBase
        :param compiler: compiler backend to use when simulating quantum circuits
        :type compiler: CompilerBase
        :param io: input/output object for saving logs, intermediate results, circuits, etc.
        :type io: IO
        :param n_hof: the size of the hall of fame (hof)
        :type n_hof: int
        :param selection_active: use selection in the evolutionary algorithm
        :type selection_active: bool
        :param save_openqasm: save population, hof, or both to openQASM strings (options: None, "hof", "pop", "both")
        :type save_openqasm: str, None
        :param noise_model_mapping: a dictionary that associates each operation type to a noise model
        :type noise_model_mapping: dict
        """

        tableau = target.stabilizer.tableau
        n_photon = tableau.n_qubits
        n_emitter = DeterministicSolver.determine_n_emitters(tableau.to_stabilizer())
        super().__init__(
            target=target,
            metric=metric,
            compiler=compiler,
            circuit=None,
            io=io,
            n_hof=n_hof,
            n_stop=n_stop,
            n_pop=n_pop,
            tournament_k=tournament_k,
            n_emitter=n_emitter,
            n_photon=n_photon,
            selection_active=selection_active,
            save_openqasm=save_openqasm,
            noise_model_mapping=noise_model_mapping,
            *args,
            **kwargs,
        )

    def initialize_transformation_probabilities(self):
        """
        Sets the initial probabilities for selecting the circuit transformations.

        :return: the transformation probabilities for possible transformations
        :rtype: dict
        """
        trans_probs = {
            self.add_emitter_one_qubit_op: 1 / 4,
            self.add_photon_one_qubit_op: 1 / 4,
            self.remove_op: 1 / 4,
            self.add_measurement_cnot_and_reset: 1 / 10,
        }

        if self.n_emitter > 1:
            trans_probs[self.add_emitter_cnot] = 1 / 4

        return self._normalize_trans_prob(trans_probs)

    def randomize_circuit(self, circuit):
        """
        Perform multiple random operations to a circuit

        :param circuit: a quantum circuit
        :type circuit: CircuitDAG
        :return: a new quantum circuit that is perturbed from the input circuit
        :rtype: CircuitDAG
        """
        randomized_circuit = circuit.copy()

        trans_probs = {
            self.add_emitter_one_qubit_op: 1 / 6,
            self.add_photon_one_qubit_op: 1 / 6,
            self.remove_op: 1 / 2,
        }

        if self.n_emitter > 1:
            trans_probs[self.add_emitter_cnot] = 1 / 6

        trans_probs = self._normalize_trans_prob(trans_probs)

        n_iter = np.random.randint(0, 10)

        for i in range(n_iter):
            transformation = np.random.choice(
                list(trans_probs.keys()), p=list(trans_probs.values())
            )
            transformation(randomized_circuit)

        return randomized_circuit

    def solve(self):
        """

        :return:
        :rtype:
        """
        deterministic_solver = DeterministicSolver(
            target=self.target,
            metric=self.metric,
            compiler=self.compiler,
        )
        deterministic_solver.noise_simulation = False
        deterministic_solver.solve()
        _, ideal_circuit = deterministic_solver.result

        # initialize the population
        population = []
        for j in range(self.n_pop):
            perturbed_circuit = self.randomize_circuit(ideal_circuit)
            population.append((np.inf, perturbed_circuit))

        self.compiler.noise_simulation = self.noise_simulation

        for i in range(self.n_stop):
            for j in range(self.n_pop):
                # choose a random transformation from allowed transformations
                transformation = np.random.choice(
                    list(self.trans_probs.keys()), p=list(self.trans_probs.values())
                )
                # evolve the chosen circuit
                circuit = population[j][1]
                transformation(circuit)
                circuit.validate()

                # compile the output state of the circuit
                compiled_state = self.compiler.compile(circuit)
                # trace out emitter qubits
                compiled_state.partial_trace(
                    keep=list(range(self.n_photon)),
                    dims=(self.n_photon + self.n_emitter) * [2],
                )
                # evaluate the metric
                score = self.metric.evaluate(compiled_state, circuit)

                population[j] = (score, circuit)

            self.update_hof(population)
            if EvolutionarySolver.use_adapt_probability:
                self.adapt_probabilities(i)

            # this should be the last thing performed *prior* to selecting a new population (after updating HoF)
            self.update_logs(population=population, iteration=i)
            self.save_circuits(population=population, hof=self.hof, iteration=i)

            if self.selection_active:
                population = self.tournament_selection(population, k=self.tournament_k)

            print(f"Iteration {i} | Best score: {self.hof[0][0]:.6f}")

        self.logs_to_df()  # convert the logs to a DataFrame
        self.result = (self.hof[0][0], self.hof[0][1])
