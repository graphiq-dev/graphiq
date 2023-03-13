import pytest
from src.circuit import CircuitDAG
from benchmarks.circuits import *
from src.ops import *
from src.utils.solver_result import *


# Test SolverResult class
def test_solver_result_init():
    circuit_1, state_1 = ghz3_state_circuit()
    circuit_2, state_2 = linear_cluster_3qubit_circuit()
    circuit_list = [circuit_1, circuit_2]

    result = SolverResult(circuit_list)

    assert len(result) == 2
    assert result["circuit"] == circuit_list


def test_solver_result_get_set_item():
    circuit_1, state_1 = ghz3_state_circuit()
    circuit_2, state_2 = linear_cluster_3qubit_circuit()
    circuit_list = [circuit_1, circuit_2]

    result = SolverResult(circuit_list)
    result["test_property"] = [1, 2]

    assert len(result) == 2
    assert len(result["test_property"]) == 2
    assert result["test_property"] == [1, 2]


def test_get_index_data():
    circuit_1, state_1 = ghz3_state_circuit()
    circuit_2, state_2 = linear_cluster_3qubit_circuit()
    circuit_list = [circuit_1, circuit_2]

    result = SolverResult(circuit_list)
    result["test_property"] = [1, 2]

    data = result.get_index_data(0)
    assert data == {
        "circuit": circuit_1,
        "test_property": 1,
    }


def test_get_circuit_index():
    circuit_1, state_1 = ghz3_state_circuit()
    circuit_2, state_2 = linear_cluster_3qubit_circuit()
    circuit_list = [circuit_1, circuit_2]

    result = SolverResult(circuit_list)
    test_circuit = result.get_circuit_index(0)

    assert test_circuit is circuit_1


def test_get_index_with_column_value():
    circuit_1, state_1 = ghz3_state_circuit()
    circuit_2, state_2 = linear_cluster_3qubit_circuit()
    circuit_3, state_3 = ghz4_state_circuit()
    circuit_list = [circuit_1, circuit_2, circuit_3]
    state_list = [state_1, state_2, state_3]

    result = SolverResult(circuit_list)
    result["state"] = state_list
    result["test_property"] = [1, 2, 1]

    assert result.get_index_with_column_value("test_property", 1) == [0, 2]
    assert result.get_index_with_column_value("test_property", 2) == [1]
