"""
Contains various hybrid solvers
"""

from src.solvers.evolutionary_solver import EvolutionarySolver
from src.solvers.deterministic_solver import DeterministicSolver
from src.backends.compiler_base import CompilerBase
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
        _, self.circuit = deterministic_solver.result
        super().solve()
