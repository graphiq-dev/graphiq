import pytest
from src.circuit.circuit_dag import CircuitDAG
from benchmarks.circuits import *
from src.utils.circuit_comparison import *
from benchmarks.graph_states import *
from src.solvers.deterministic_solver import DeterministicSolver
from src.solvers.hybrid_solvers import (
    HybridGraphSearchSolver,
    HybridGraphSearchSolverSetting,
)
from src.metrics import Infidelity
from src.state import QuantumState
from src.backends.stabilizer.compiler import StabilizerCompiler
from src.backends.stabilizer.functions.rep_conversion import (
    get_clifford_tableau_from_graph,
)
from src.utils.solver_result import SolverResult
import numpy as np


def get_pipeline(target_graph):
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

    return solver


# Test circuit_is_isomorphic()
def test_circuit_is_isomorphic_ghz3():
    circuit1, state1 = ghz3_state_circuit()
    circuit2, state2 = ghz3_state_circuit()

    assert circuit_is_isomorphic(circuit1, circuit2)


@pytest.mark.parametrize(
    "circuit1_params, circuit2_params, result",
    [
        ((1, 1), (1, 1), True),
        ((1, 2), (2, 1), False),
        ((3, 2), (3, 2), True),
    ],
)
def test_circuit_is_isomorphic_empty(circuit1_params, circuit2_params, result):
    circuit1 = CircuitDAG(n_photon=circuit1_params[0], n_emitter=circuit1_params[1])
    circuit2 = CircuitDAG(n_photon=circuit2_params[0], n_emitter=circuit2_params[1])

    assert circuit_is_isomorphic(circuit1, circuit2) == result


def test_circuit_is_isomorphic_1():
    circuit1 = CircuitDAG(n_photon=2)
    circuit2 = CircuitDAG(n_photon=2)

    circuit1.add(Hadamard(reg_type="p", register=0))
    circuit2.add(Hadamard(reg_type="p", register=1))

    assert circuit_is_isomorphic(circuit1, circuit2)


def test_circuit_is_isomorphic_2():
    circuit1 = CircuitDAG(n_emitter=2)
    circuit2 = CircuitDAG(n_emitter=2)

    circuit1.add(Hadamard(reg_type="e", register=0))
    circuit2.add(Hadamard(reg_type="e", register=1))

    assert circuit_is_isomorphic(circuit1, circuit2)


def test_circuit_is_isomorphic_3():
    # swap photons
    circuit_1 = CircuitDAG(n_emitter=1, n_photon=3)
    circuit_1.add(CNOT(0, "e", 0, "p"))
    circuit_1.add(CNOT(0, "e", 1, "p"))
    circuit_1.add(CNOT(0, "e", 2, "p"))

    circuit_2 = CircuitDAG(n_emitter=1, n_photon=3)
    circuit_2.add(CNOT(0, "e", 0, "p"))
    circuit_2.add(CNOT(0, "e", 2, "p"))
    circuit_2.add(CNOT(0, "e", 1, "p"))

    assert circuit_is_isomorphic(circuit_1, circuit_2)


def test_circuit_is_isomorphic_4():
    # %% 2 emitters and 2 photons, swap only emitters
    circuit_1 = CircuitDAG(n_emitter=2, n_photon=1, n_classical=0)
    circuit_1.add(Hadamard(register=0, reg_type="e"))
    circuit_1.add(Hadamard(register=1, reg_type="e"))
    circuit_1.add(CNOT(control=0, control_type="e", target=0, target_type="p"))
    circuit_1.add(CNOT(control=0, control_type="e", target=1, target_type="e"))
    circuit_1.add(CNOT(control=1, control_type="e", target=1, target_type="p"))
    circuit_1.validate()

    circuit_2 = CircuitDAG(n_emitter=2, n_photon=1, n_classical=0)
    circuit_2.add(Hadamard(register=1, reg_type="e"))
    circuit_2.add(Hadamard(register=0, reg_type="e"))
    circuit_2.add(CNOT(control=1, control_type="e", target=0, target_type="p"))
    circuit_2.add(CNOT(control=1, control_type="e", target=0, target_type="e"))
    circuit_2.add(CNOT(control=0, control_type="e", target=1, target_type="p"))
    circuit_2.validate()

    assert circuit_is_isomorphic(circuit_1, circuit_2)


def test_circuit_is_isomorphic_5():
    # %% 2 emitters and 2 photons, swap all emitters and photons
    circuit_1 = CircuitDAG(n_emitter=2, n_photon=1, n_classical=0)
    circuit_1.add(Hadamard(register=0, reg_type="e"))
    circuit_1.add(Hadamard(register=1, reg_type="e"))
    circuit_1.add(CNOT(control=0, control_type="e", target=0, target_type="p"))
    circuit_1.add(CNOT(control=0, control_type="e", target=1, target_type="e"))
    circuit_1.add(CNOT(control=1, control_type="e", target=1, target_type="p"))

    circuit_2 = CircuitDAG(n_emitter=2, n_photon=1, n_classical=0)
    circuit_2.add(Hadamard(register=1, reg_type="e"))
    circuit_2.add(Hadamard(register=0, reg_type="e"))
    circuit_2.add(CNOT(control=1, control_type="e", target=1, target_type="p"))
    circuit_2.add(CNOT(control=1, control_type="e", target=0, target_type="e"))
    circuit_2.add(CNOT(control=0, control_type="e", target=0, target_type="p"))

    assert circuit_is_isomorphic(circuit_1, circuit_2)


def test_circuit_is_isomorphic_6():
    # Same dag but different control target on node
    circuit_1 = CircuitDAG(n_emitter=2, n_photon=1, n_classical=0)
    circuit_1.add(Hadamard(register=0, reg_type="e"))
    circuit_1.add(Hadamard(register=1, reg_type="e"))
    circuit_1.add(CNOT(control=0, control_type="e", target=0, target_type="p"))
    circuit_1.add(CNOT(control=0, control_type="e", target=1, target_type="e"))
    circuit_1.add(CNOT(control=1, control_type="e", target=1, target_type="p"))

    circuit_2 = CircuitDAG(n_emitter=2, n_photon=1, n_classical=0)
    circuit_2.add(Hadamard(register=0, reg_type="e"))
    circuit_2.add(Hadamard(register=1, reg_type="e"))
    circuit_2.add(CNOT(control=0, control_type="e", target=0, target_type="p"))
    circuit_2.add(CNOT(control=1, control_type="e", target=0, target_type="e"))
    circuit_2.add(CNOT(control=1, control_type="e", target=1, target_type="p"))

    assert not circuit_is_isomorphic(circuit_1, circuit_2)


def test_circuit_is_isomorphic_7():
    # swap e0 -> e1, e1 -> e2
    circuit_1 = CircuitDAG(n_emitter=3, n_photon=3, n_classical=0)
    circuit_1.add(Hadamard(register=0, reg_type="e"))
    circuit_1.add(Hadamard(register=1, reg_type="e"))
    circuit_1.add(Hadamard(register=2, reg_type="e"))
    circuit_1.add(CNOT(control=0, control_type="e", target=0, target_type="p"))
    circuit_1.add(CNOT(control=0, control_type="e", target=1, target_type="e"))
    circuit_1.add(CNOT(control=1, control_type="e", target=1, target_type="p"))
    circuit_1.add(CNOT(control=1, control_type="e", target=2, target_type="e"))
    circuit_1.add(CNOT(control=2, control_type="e", target=2, target_type="p"))

    circuit_2 = CircuitDAG(n_emitter=3, n_photon=3, n_classical=0)
    circuit_2.add(Hadamard(register=0, reg_type="e"))
    circuit_2.add(Hadamard(register=1, reg_type="e"))
    circuit_2.add(Hadamard(register=2, reg_type="e"))
    circuit_2.add(CNOT(control=1, control_type="e", target=0, target_type="p"))
    circuit_2.add(CNOT(control=1, control_type="e", target=2, target_type="e"))
    circuit_2.add(CNOT(control=2, control_type="e", target=1, target_type="p"))
    circuit_2.add(CNOT(control=2, control_type="e", target=0, target_type="e"))
    circuit_2.add(CNOT(control=0, control_type="e", target=2, target_type="p"))

    assert circuit_is_isomorphic(circuit_1, circuit_2)


def test_circuit_is_isomorphic_8():
    # swap e0 -> e1, e1 -> e2
    # Same dag but different control target on node
    circuit_1 = CircuitDAG(n_emitter=3, n_photon=3, n_classical=0)
    circuit_1.add(Hadamard(register=0, reg_type="e"))
    circuit_1.add(Hadamard(register=1, reg_type="e"))
    circuit_1.add(Hadamard(register=2, reg_type="e"))
    circuit_1.add(CNOT(control=0, control_type="e", target=0, target_type="p"))
    circuit_1.add(CNOT(control=0, control_type="e", target=1, target_type="e"))
    circuit_1.add(CNOT(control=1, control_type="e", target=1, target_type="p"))
    circuit_1.add(CNOT(control=1, control_type="e", target=2, target_type="e"))
    circuit_1.add(CNOT(control=2, control_type="e", target=2, target_type="p"))

    circuit_2 = CircuitDAG(n_emitter=3, n_photon=3, n_classical=0)
    circuit_2.add(Hadamard(register=0, reg_type="e"))
    circuit_2.add(Hadamard(register=1, reg_type="e"))
    circuit_2.add(Hadamard(register=2, reg_type="e"))
    circuit_2.add(CNOT(control=1, control_type="e", target=0, target_type="p"))
    circuit_2.add(CNOT(control=2, control_type="e", target=1, target_type="e"))
    circuit_2.add(CNOT(control=2, control_type="e", target=1, target_type="p"))
    circuit_2.add(CNOT(control=2, control_type="e", target=0, target_type="e"))
    circuit_2.add(CNOT(control=0, control_type="e", target=2, target_type="p"))

    assert not circuit_is_isomorphic(circuit_1, circuit_2)


# Test remove_redundant_circuits()
def test_remove_redundant_circuits_1():
    circuit_1 = CircuitDAG(n_emitter=2, n_photon=1, n_classical=0)
    circuit_1.add(Hadamard(register=0, reg_type="e"))
    circuit_1.add(Hadamard(register=1, reg_type="e"))
    circuit_1.add(CNOT(control=0, control_type="e", target=0, target_type="p"))
    circuit_1.add(CNOT(control=0, control_type="e", target=1, target_type="e"))
    circuit_1.add(CNOT(control=1, control_type="e", target=1, target_type="p"))

    circuit_2 = CircuitDAG(n_emitter=2, n_photon=1, n_classical=0)
    circuit_2.add(Hadamard(register=1, reg_type="e"))
    circuit_2.add(Hadamard(register=0, reg_type="e"))
    circuit_2.add(CNOT(control=1, control_type="e", target=0, target_type="p"))
    circuit_2.add(CNOT(control=1, control_type="e", target=0, target_type="e"))
    circuit_2.add(CNOT(control=0, control_type="e", target=1, target_type="p"))
    circuit_2.validate()

    circuit_list = [circuit_1, circuit_2]
    new_list = remove_redundant_circuits(circuit_list)

    assert len(new_list) == 1


# Test circuit equivalency with same circuit properties
# Test with pipeline
def test_circuit_equivalency_1():
    # Test with pipeline, with property: n_emitters
    target_graph = linear_cluster_state(4)
    target_graph = target_graph.data

    solver = get_pipeline(target_graph)
    circuit_data = solver.solve()

    circuit_list = []
    max_emitter_depth_list = []
    n_emitter_list = []

    for data in circuit_data:
        circuit = data[0]
        n_emitter = circuit.n_emitters
        n_emitter_list.append(n_emitter)

        circuit_list.append(circuit)

        max_emitter_depth = max(circuit.calculate_reg_depth("e"))
        max_emitter_depth_list.append(max_emitter_depth)

    result = SolverResult(circuit_list)
    result["max_emitter_depth"] = max_emitter_depth_list
    result["n_emitters"] = n_emitter_list

    for i in np.unique(result["n_emitters"]):
        index = result.get_index_with_column_value("n_emitters", i)
        circuit_list = []
        for j in index:
            new_circuit = result.get_circuit_index(j)
            if circuit_list:
                for circuit in circuit_list:
                    assert not check_redundant_circuit(circuit, new_circuit)
            else:
                circuit_list.append(new_circuit)


def test_circuit_equivalency_2():
    # Test with pipeline, with property: max_emitter_depth
    target_graph = linear_cluster_state(4)
    target_graph = target_graph.data

    solver = get_pipeline(target_graph)
    circuit_data = solver.solve()

    circuit_list = []
    max_emitter_depth_list = []
    n_emitter_list = []

    for data in circuit_data:
        circuit = data[0]
        n_emitter = circuit.n_emitters
        n_emitter_list.append(n_emitter)

        circuit_list.append(circuit)

        max_emitter_depth = max(circuit.calculate_reg_depth("e"))
        max_emitter_depth_list.append(max_emitter_depth)

    result = SolverResult(circuit_list)
    result["max_emitter_depth"] = max_emitter_depth_list
    result["n_emitters"] = n_emitter_list

    for i in np.unique(result["max_emitter_depth"]):
        index = result.get_index_with_column_value("max_emitter_depth", i)
        circuit_list = []
        for j in index:
            new_circuit = result.get_circuit_index(j)
            if circuit_list:
                for circuit in circuit_list:
                    assert not check_redundant_circuit(circuit, new_circuit)
            else:
                circuit_list.append(new_circuit)


# Test outside the pipeline
def test_circuit_equivalency_3():
    # Same circuit emitter depth, but different in operations
    circuit_1 = CircuitDAG(n_emitter=2, n_photon=1, n_classical=0)
    circuit_1.add(Hadamard(register=0, reg_type="e"))
    circuit_1.add(Hadamard(register=1, reg_type="e"))
    circuit_1.add(CNOT(control=0, control_type="e", target=0, target_type="p"))
    circuit_1.add(CNOT(control=0, control_type="e", target=1, target_type="e"))
    circuit_1.add(CNOT(control=1, control_type="e", target=1, target_type="p"))

    circuit_2 = CircuitDAG(n_emitter=2, n_photon=1, n_classical=0)
    circuit_2.add(Hadamard(register=1, reg_type="e"))
    circuit_2.add(SigmaZ(register=0, reg_type="e"))
    circuit_2.add(CNOT(control=1, control_type="e", target=0, target_type="p"))
    circuit_2.add(CNOT(control=1, control_type="e", target=0, target_type="e"))
    circuit_2.add(CNOT(control=0, control_type="e", target=1, target_type="p"))

    assert sorted(circuit_1.calculate_reg_depth("e")) == sorted(
        circuit_2.calculate_reg_depth("e")
    )
    assert not check_redundant_circuit(circuit_1, circuit_2)


def test_circuit_equivalency_4():
    # Same circuit emitter depth, but add Identity
    circuit_1, state_1 = ghz3_state_circuit()
    circuit_2, state_2 = ghz3_state_circuit()

    circuit_1.add(Identity(register=0, reg_type="p"))

    assert sorted(circuit_1.calculate_reg_depth("e")) == sorted(
        circuit_2.calculate_reg_depth("e")
    )
    assert check_redundant_circuit(circuit_1, circuit_2)


# Test CircuitStorage class
def test_circuit_storage_init():
    # Test CircuitStorage init
    storage = CircuitStorage()

    assert storage.circuit_list == []
    assert not storage.disable_circuit_comparison


def test_circuit_storage_add_new_circuit():
    # Test CircuitStorage.add_new_circuit function
    storage = CircuitStorage()

    circuit_1, state_1 = ghz3_state_circuit()
    circuit_2, state_2 = linear_cluster_3qubit_circuit()

    check_1 = storage.add_new_circuit(circuit_1)
    check_2 = storage.add_new_circuit(circuit_2)

    assert len(storage.circuit_list) == 2
    assert check_1
    assert check_2


def test_circuit_storage_is_redundant():
    # Test CircuitStorage.add_new_circuit function with same circuit add
    storage = CircuitStorage()

    circuit_1, state_1 = ghz3_state_circuit()
    circuit_2, state_2 = linear_cluster_3qubit_circuit()
    circuit_3, state_3 = linear_cluster_3qubit_circuit()

    check_1 = storage.add_new_circuit(circuit_1)
    check_2 = storage.add_new_circuit(circuit_2)
    check_3 = storage.add_new_circuit(circuit_3)

    assert check_1
    assert check_2
    assert not check_3
    assert len(storage.circuit_list) == 2
