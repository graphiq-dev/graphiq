"""
Contains various hybrid solvers

"""

from src.solvers.evolutionary_solver import EvolutionarySolver
from src.solvers.deterministic_solver import DeterministicSolver
from src.backends.compiler_base import CompilerBase
from src.metrics import MetricBase
from src.io import IO


class HybridEvolutionarySolver(EvolutionarySolver):
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
        deterministic_solver = DeterministicSolver(
            target=self.target,
            metric=self.metric,
            compiler=self.compiler,
        )
        deterministic_solver.noise_simulation = False
        deterministic_solver.solve()
        _, self.circuit = deterministic_solver.result
        super().solve()
