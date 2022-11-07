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
from src.backends.stabilizer.compiler import StabilizerCompiler
from src.utils.circuit_comparison import compare_circuits
import src.noise.noise_models as noise


def search_for_alternative_circuits(
    graph, noise_model_mapping, metric_class, random_seed
):
    """
    Run arbitrary input graph state with noise simulation using metric

    :param graph:
    :type graph:
    :param noise_model_mapping:
    :type noise_model_mapping:
    :param metric_class:
    :type metric_class:
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

    benchmark_score, benchmark_circuit = det_solver.result
    results.append(benchmark_circuit)
    hybrid_solver = HybridEvolutionarySolver(
        target=target,
        metric=metric,
        compiler=compiler,
        noise_model_mapping=noise_model_mapping,
    )
    hybrid_solver.seed(random_seed)
    hybrid_solver.solve()

    for i in range(hybrid_solver.setting.n_hof):
        alternate_score = hybrid_solver.hof[i][0]
        alternate_circuit = hybrid_solver.hof[i][1]
        if alternate_score <= benchmark_score and not compare_circuits(
            benchmark_circuit, alternate_circuit
        ):
            results.append(alternate_circuit)
    return results


def run_one_repeater_graph_state(n_inner_qubits, random_seed):
    """
    Run deterministic solver to get a benchmark circuit and then run the hybrid solver. Check if any alternate circuits
    are found.
    No noise simulation

    :param n_inner_qubits:
    :type n_inner_qubits:
    :param random_seed:
    :type random_seed:
    :return:
    :rtype:
    """
    graph = repeater_graph_states(n_inner_qubits)
    return search_for_alternative_circuits(graph, None, Infidelity, random_seed)


def run_one_repeater_graph_state_w_loss(
    n_inner_qubits, error_rate, loss_rate, random_seed
):
    """
    Run deterministic solver to get a benchmark circuit and then run the hybrid solver. Check if any alternate circuits
    are found.

    :param n_inner_qubits:
    :type n_inner_qubits:
    :param error_rate:
    :type error_rate:
    :param loss_rate:
    :type loss_rate:
    :param random_seed:
    :type random_seed:
    :return:
    :rtype:
    """
    graph = repeater_graph_states(n_inner_qubits)
    emitter_noise = noise.DepolarizingNoise(error_rate)
    photon_loss = noise.PhotonLoss(loss_rate)
    noise_model_mapping = {
        "e": {},
        "p": {"OneQubitGateWrapper": photon_loss},
        "ee": {},
        "ep": {},
    }
    return search_for_alternative_circuits(
        graph, noise_model_mapping, Infidelity, random_seed
    )
