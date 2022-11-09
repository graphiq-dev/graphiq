"""
A script to find alternative circuits with hybrid solver
"""

from src.solvers.hybrid_solvers import HybridEvolutionarySolver
from src.solvers.deterministic_solver import DeterministicSolver
from benchmarks.graph_states import *
from src.metrics import Infidelity
from src.backends.stabilizer.functions.rep_conversion import (
    get_clifford_tableau_from_graph,
)
from src.state import QuantumState
from src.backends.stabilizer.state import MixedStabilizer
from src.backends.stabilizer.compiler import StabilizerCompiler
from src.utils.circuit_comparison import compare_circuits


def search_for_alternative_circuits(
    graph, noise_model_mapping, metric_class, solver_setting, random_seed=1
):
    """
    Run arbitrary input graph state with noise simulation using the given metric. It first calls the deterministic
    solver to get a benchmark circuit and score. It then calls the hybrid solver based on deterministic solver and
    random / evolutionary search solver

    :param graph:
    :type graph:
    :param noise_model_mapping:
    :type noise_model_mapping:
    :param metric_class:
    :type metric_class:
    :param solver_setting:
    :type solver_setting:
    :param random_seed:
    :type random_seed:
    :return:
    :rtype:
    """
    results = []
    target_tableau = get_clifford_tableau_from_graph(graph)
    n_photon = target_tableau.n_qubits
    target = QuantumState(n_photon, target_tableau, representation="stabilizer")
    compiler = StabilizerCompiler()
    compiler.measurement_determinism = 1
    metric = metric_class(target)
    det_solver = DeterministicSolver(
        target=target,
        metric=metric,
        compiler=compiler,
        noise_model_mapping=noise_model_mapping,
    )
    det_solver.solve()
    n_emitter = det_solver.n_emitter
    benchmark_score, benchmark_circuit = det_solver.result
    compiler.noise_simulation = True
    compiled_state = compiler.compile(benchmark_circuit)
    # trace out emitter qubits
    compiled_state.partial_trace(
        keep=list(range(n_photon)),
        dims=(n_photon + n_emitter) * [2],
    )
    if isinstance(compiled_state.stabilizer, MixedStabilizer):
        prob = compiled_state.stabilizer.probability
    else:
        prob = 1

    print("starting hybrid solver")
    results.append((benchmark_score, prob, benchmark_circuit))
    hybrid_solver = HybridEvolutionarySolver(
        target=target,
        metric=metric,
        compiler=compiler,
        noise_model_mapping=noise_model_mapping,
        solver_setting=solver_setting,
    )
    hybrid_solver.seed(random_seed)
    hybrid_solver.solve()

    for i in range(hybrid_solver.setting.n_hof):
        alternate_score = hybrid_solver.hof[i][0]
        alternate_circuit = hybrid_solver.hof[i][1]

        if alternate_score <= benchmark_score and not compare_circuits(
            benchmark_circuit, alternate_circuit
        ):
            compiler.noise_simulation = True
            compiled_state = compiler.compile(alternate_circuit)
            # trace out emitter qubits
            compiled_state.partial_trace(
                keep=list(range(n_photon)),
                dims=(n_photon + n_emitter) * [2],
            )
            if isinstance(compiled_state.stabilizer, MixedStabilizer):
                prob = compiled_state.stabilizer.probability
            else:
                prob = 1
            results.append((alternate_score, prob, alternate_circuit))
    return results


def report_alternate_circuits(results):
    if len(results) > 1:
        print(f"Find {len(results)} circuits that produce the same state.")
        print(
            f"The circuit (first one drawn) returned by the deterministic solver has a score of {results[1][0]},\
            and probability of not losing any photon: {results[1][1]}"
        )
        results[0][2].draw_circuit()
        for i in range(1, len(results)):
            print(
                f"Circuit {i} has a score of {results[i][0]}, and a probability of not losing any photon: {results[i][1]}"
            )
            results[i][2].draw_circuit()
