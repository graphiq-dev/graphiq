"""
A script to find alternative circuits with hybrid solver
"""

from src.solvers.hybrid_solvers import HybridEvolutionarySolver
from src.solvers.deterministic_solver import DeterministicSolver
from src.backends.stabilizer.functions.rep_conversion import (
    get_clifford_tableau_from_graph,
)
from src.state import QuantumState
from src.backends.stabilizer.state import MixedStabilizer
from src.backends.stabilizer.compiler import StabilizerCompiler
from src.utils.circuit_comparison import compare_circuits
from src.metrics import Infidelity
from src.solvers.evolutionary_solver import EvolutionarySearchSolverSetting
import src.noise.noise_models as noise


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
    compiler.noise_simulation = True
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

    results.append((benchmark_score, prob, benchmark_circuit))
    print("starting hybrid solver")
    hybrid_solver = HybridEvolutionarySolver(
        target=target,
        metric=metric,
        compiler=compiler,
        noise_model_mapping=noise_model_mapping,
        solver_setting=solver_setting,
    )
    hybrid_solver.seed(random_seed)
    hybrid_solver.solve()
    tolerance = 0.001
    for i in range(hybrid_solver.setting.n_hof):
        alternate_score = hybrid_solver.hof[i][0]
        alternate_circuit = hybrid_solver.hof[i][1]

        if alternate_score <= benchmark_score + tolerance and not compare_circuits(
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
    print(f"Find {len(results)} circuits that produce the same state.")
    print(
        f"The circuit (first one drawn) returned by the deterministic solver has a score of {results[0][0]},\
               and probability of not losing any photon: {results[0][1]}"
    )
    results[0][2].draw_circuit()
    if len(results) > 1:
        for i in range(1, len(results)):
            print(
                f"Circuit {i} has a score of {results[i][0]}, and a probability of not losing any photon: {results[i][1]}"
            )
            results[i][2].draw_circuit()
    else:
        print("There is no alternative circuit found.")


def noise_model_loss_and_depolarizing(error_rate, loss_rate):
    emitter_noise = noise.DepolarizingNoise(error_rate)
    photon_loss = noise.PhotonLoss(loss_rate)
    noise_model_mapping = {
        "e": {"CNOT": emitter_noise},
        "p": {"Hadamard": photon_loss, "OneQubitGateWrapper": photon_loss},
        "ee": {"CNOT_control": emitter_noise, "CNOT_target": emitter_noise},
        "ep": {"CNOT_control": emitter_noise},
    }
    return noise_model_mapping


def noise_model_pauli_error():
    pauli_x_error = noise.PauliError("X")
    pauli_y_error = noise.PauliError("Y")
    pauli_z_error = noise.PauliError("Z")
    noise_model_mapping = {
        "e": {
            "SigmaX": pauli_y_error,
            "SigmaY": pauli_z_error,
            "SigmaZ": pauli_x_error,
        },
        "p": {
            "SigmaX": pauli_y_error,
            "SigmaY": pauli_z_error,
            "SigmaZ": pauli_x_error,
        },
        "ee": {},
        "ep": {},
    }
    return noise_model_mapping


def noise_model_pure_loss(loss_rate):
    photon_loss = noise.PhotonLoss(loss_rate)
    noise_model_mapping = {
        "e": {},
        "p": {
            "Hadamard": photon_loss,
            "Phase": photon_loss,
            "SigmaX": photon_loss,
            "SigmaY": photon_loss,
            "SigmaZ": photon_loss,
            "OneQubitGateWrapper": photon_loss,
        },
        "ee": {},
        "ep": {},
    }
    return noise_model_mapping


def exemplary_test(graph, noise_model_mapping, solver_setting=None, random_seed=1):
    if solver_setting is None:
        solver_setting = EvolutionarySearchSolverSetting()
    results = search_for_alternative_circuits(
        graph, noise_model_mapping, Infidelity, solver_setting, random_seed
    )
    report_alternate_circuits(results)


def exemplary_multiple_test(
    graph, noise_model_mapping, random_numbers, solver_setting=None
):
    if solver_setting is None:
        solver_setting = EvolutionarySearchSolverSetting()
    for i in range(len(random_numbers)):
        results = search_for_alternative_circuits(
            graph,
            noise_model_mapping,
            Infidelity,
            solver_setting,
            random_seed=random_numbers[i],
        )
        if len(results) > 1:
            print(f"The random seed that works is {random_numbers[i]}.")
            break
    report_alternate_circuits(results)
