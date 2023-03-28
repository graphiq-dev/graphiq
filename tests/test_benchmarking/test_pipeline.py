from benchmarks.pipeline import *
from src.metrics import Infidelity
from src.io import IO
from src.solvers.deterministic_solver import DeterministicSolver
from src.solvers.evolutionary_solver import EvolutionarySearchSolverSetting
from src.solvers.hybrid_solvers import (
    HybridEvolutionarySolver,
    HybridGraphSearchSolver,
    HybridGraphSearchSolverSetting,
)
from src.backends.stabilizer.compiler import StabilizerCompiler
from src.backends.stabilizer.functions.rep_conversion import (
    get_clifford_tableau_from_graph,
)
from src.state import QuantumState
from benchmarks.graph_states import *


def test_benchmark_run_graph_search_solver():
    target_graph = linear_cluster_state(4)
    target_graph = target_graph.data
    target_tableau = get_clifford_tableau_from_graph(target_graph)
    n_photon = target_tableau.n_qubits
    target_state = QuantumState(n_photon, target_tableau, representation="stabilizer")

    compiler = StabilizerCompiler()
    metric = Infidelity(target=target_state)
    solver_setting = HybridGraphSearchSolverSetting(n_iso_graphs=5, n_lc_graphs=5)

    solver = HybridGraphSearchSolver(
        target=target_state,
        metric=metric,
        compiler=compiler,
        graph_solver_setting=solver_setting,
        base_solver=DeterministicSolver,
    )

    example_run = {
        "solver": solver,
        "target": target_state,
        "metric": metric,
        "compiler": compiler,
    }
    io = IO.new_directory(
        folder="benchmarks", include_date=True, include_time=True, include_id=False
    )

    data = benchmark_run_graph_search_solver(example_run, "test_pipeline", io)
    assert type(data) == dict
    assert data.keys() == ["name", "path", "solver", "compiler", "metric", "time"]
