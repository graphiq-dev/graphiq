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
from src.solvers.evolutionary_solver import EvolutionarySolverSetting
import src.noise.noise_models as noise


def search_for_alternative_circuits(
    graph, noise_model_mapping, metric_class, solver_setting, random_seed=1
):
    """
    Run arbitrary input graph state with noise simulation using the given metric. It first calls the deterministic
    solver to get a benchmark circuit and score. It then calls the hybrid solver based on deterministic solver and
    random / evolutionary search solver

    :param graph: a graph that represents a graph state
    :type graph: networkX.Graph
    :param noise_model_mapping: a way to assign noises to gates
    :type noise_model_mapping: dict
    :param metric_class: a metric class
    :type metric_class: MetricBase or its subclass
    :param solver_setting: specifies the setting of a chosen solver
    :type solver_setting: RandomSearchSolverSetting or EvolutionarySolverSetting
    :param random_seed: a random seed
    :type random_seed: int
    :return: results are stored in terms of (score, survival probability, circuit)
    :rtype: list[tuple(float, float, CircuitDAG)]
    """
    results = []
    target_tableau = get_clifford_tableau_from_graph(graph)
    n_photon = target_tableau.n_qubits
    target = QuantumState(target_tableau, rep_type="s")
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
    if isinstance(compiled_state.rep_data, MixedStabilizer):
        prob = compiled_state.rep_data.probability

    else:
        prob = 1

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
    # add a small tolerance for score comparison
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
            if isinstance(compiled_state.rep_data, MixedStabilizer):
                prob = compiled_state.rep_data.probability
            else:
                prob = 1
            results.append((alternate_score, prob, alternate_circuit))
    return results


def report_alternate_circuits(results):
    """
    A way to print out results

    :param results: results are stored in terms of (score, survival probability, circuit)
    :type results: tuple
    :return: nothing
    :rtype: None
    """
    print(f"Find {len(results)} circuits that produce the same state.")
    print(
        f"The circuit (first one drawn) returned by the deterministic solver has a score of {results[0][0]},\
               and probability of not losing any photon: {results[0][1]}"
    )
    circuit = results[0][2]
    circuit.draw_circuit()
    emitter_depth = circuit.calculate_reg_depth("e")
    print(f"emitter depths of circuit 1 are {emitter_depth}")
    if len(results) > 1:
        for i in range(1, len(results)):
            print(
                f"Circuit {i} has a score of {results[i][0]}, and a probability of not losing any photon: {results[i][1]}"
            )
            alt_circuit = results[i][2]
            alt_circuit.draw_circuit()
            emitter_depth = alt_circuit.calculate_reg_depth("e")
            print(f"emitter depths of circuit {i} are {emitter_depth}")
    else:
        print("There is no alternative circuit found.")


def noise_model_loss_and_depolarizing(depolarizing_prob, loss_rate):
    """
    An example of noise model with photon losses and depolarizing noises

    :param depolarizing_prob: the depolarizing probability
    :type depolarizing_prob: float
    :param loss_rate: the probability of losing one photon
    :type loss_rate: float
    :return: a dictionary that specifies the noise model with photon losses and depolarizing noises
    :rtype: dict
    """
    emitter_noise = noise.DepolarizingNoise(depolarizing_prob)
    photon_loss = noise.PhotonLoss(loss_rate)
    noise_model_mapping = {
        "e": {},
        "p": {"Hadamard": photon_loss, "OneQubitGateWrapper": photon_loss},
        "ee": {"CNOT_control": emitter_noise, "CNOT_target": emitter_noise},
        "ep": {"CNOT_control": emitter_noise},
    }
    return noise_model_mapping


def noise_model_pauli_error():
    """
    An example of noise model with only Pauli errors

    :return: a way to map noises to gates
    :rtype: dict
    """
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
    """
    A noise model with only photon losses

    :param loss_rate: the probability of losing one photon
    :type loss_rate: float
    :return: a way to map noises to gates
    :rtype: dict
    """
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


def exemplary_run(graph, noise_model_mapping, solver_setting=None, random_seed=1):
    """
    Run one exemplary test to search alternative circuits for an input graph state.

    :param graph: an input graph that represents a graph state
    :type graph: networkX.Graph
    :param noise_model_mapping: a way to map noises to gates
    :type noise_model_mapping: dict
    :param solver_setting: specifies the setting of the chosen solver
    :type solver_setting: RandomSearchSolverSetting or EvolutionarySolverSetting
    :param random_seed: a random seed
    :type random_seed: int
    :return: results are stored in terms of (score, survival probability, circuit)
    :rtype: list[tuple(float, float, CircuitDAG)]
    """
    if solver_setting is None:
        solver_setting = EvolutionarySolverSetting()
    results = search_for_alternative_circuits(
        graph, noise_model_mapping, Infidelity, solver_setting, random_seed
    )
    return results


def exemplary_test(graph, noise_model_mapping, solver_setting=None, random_seed=1):
    """
    Run one exemplary test to search alternative circuits for an input graph state.

    :param graph: an input graph that represents a graph state
    :type graph: networkX.Graph
    :param noise_model_mapping: a way to map noises to gates
    :type noise_model_mapping: dict
    :param solver_setting: specifies the setting of the chosen solver
    :type solver_setting: RandomSearchSolverSetting or EvolutionarySolverSetting
    :param random_seed: a random seed
    :type random_seed: int
    :return: nothing
    :rtype: None
    """
    results = exemplary_run(graph, noise_model_mapping, solver_setting, random_seed)
    report_alternate_circuits(results)


def exemplary_multiple_test(
    graph, noise_model_mapping, random_numbers, solver_setting=None
):
    """
    Run exemplary test multiple times

    :param graph: an input graph that represents a graph state
    :type graph: networkX.Graph
    :param noise_model_mapping: a way to map noises to gates
    :type noise_model_mapping: dict
    :param random_numbers: a list of random numbers
    :type random_numbers: list[int]
    :param solver_setting: specifies the setting of the chosen solver
    :type solver_setting: RandomSearchSolverSetting or EvolutionarySolverSetting
    :return: nothing
    :rtype: None
    """
    if solver_setting is None:
        solver_setting = EvolutionarySolverSetting()
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
