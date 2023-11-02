import numpy as np
import numpy.testing as test

import graphiq.circuit.ops as ops
from graphiq.circuit.circuit_dag import CircuitDAG


# Test initialization
# Test registers depth after circuit initialization
# Better with @pytest.mark.parametrize for multiple cases check
def test_initialization():
    circuit = CircuitDAG(n_emitter=1, n_photon=4, n_classical=1)
    expected_depth = {
        "e": np.array([0]),
        "p": np.array([0, 0, 0, 0]),
        "c": np.array([0]),
    }

    test.assert_equal(circuit.register_depth, expected_depth)


# Test add operation 1
# Test registers depth after add new operation to registers
def test_add_operation1():
    circuit = CircuitDAG(n_emitter=1, n_photon=2, n_classical=1)
    expected_depth = {"e": np.array([0]), "p": np.array([1, 0]), "c": np.array([0])}

    circuit.add(ops.SigmaX(register=0, reg_type="p"))
    circuit.calculate_all_reg_depth()

    test.assert_equal(circuit.register_depth, expected_depth)


# Test add operation 2
# Test registers depth after add CNOT gate to registers
def test_add_operation2():
    circuit = CircuitDAG(n_emitter=1, n_photon=1, n_classical=1)
    expected_depth = {"e": np.array([4]), "p": np.array([4]), "c": np.array([0])}

    circuit.add(ops.SigmaX(register=0, reg_type="p"))
    circuit.add(ops.Hadamard(register=0, reg_type="p"))
    circuit.add(ops.SigmaX(register=0, reg_type="p"))
    circuit.add(ops.CNOT(control=0, control_type="e", target_type="p", target=0))
    circuit.calculate_all_reg_depth()

    test.assert_equal(circuit.register_depth, expected_depth)


# Test insert operation at
# Test registers depth after insert new operation at nodes
def test_insert_operation_at():
    circuit = CircuitDAG(n_emitter=1, n_photon=1, n_classical=1)
    expected_depth = {"e": [5], "p": [5], "c": [0]}

    circuit.add(ops.SigmaX(register=0, reg_type="p"))
    circuit.add(ops.Hadamard(register=0, reg_type="p"))
    circuit.add(ops.SigmaX(register=0, reg_type="p"))
    circuit.add(ops.CNOT(control=0, control_type="e", target_type="p", target=0))
    circuit.insert_at(ops.SigmaY(register=0, reg_type="p"), edges=[("p0_in", 1, "p0")])
    circuit.calculate_all_reg_depth()

    test.assert_equal(circuit.register_depth, expected_depth)


# Test replace operation
# Test registers depth after replace operation at nodes
def test_replace_operation():
    circuit = CircuitDAG(n_emitter=1, n_photon=1, n_classical=1)
    expected_depth = {"e": [0], "p": [1], "c": [0]}

    circuit.add(ops.SigmaX(register=0, reg_type="p"))
    circuit.replace_op(node=1, new_operation=ops.SigmaY(register=0, reg_type="p"))
    circuit.calculate_all_reg_depth()

    test.assert_equal(circuit.register_depth, expected_depth)


# Test remove operation
# Test registers depth after remove operation at nodes
def test_remove_operation():
    circuit = CircuitDAG(n_emitter=1, n_photon=1, n_classical=1)
    expected_depth = {"e": [0], "p": [2], "c": [0]}

    circuit.add(ops.SigmaX(register=0, reg_type="p"))
    circuit.add(ops.Hadamard(register=0, reg_type="p"))
    circuit.add(ops.SigmaX(register=0, reg_type="p"))
    circuit.remove_op(node=3)
    circuit.calculate_all_reg_depth()

    test.assert_equal(circuit.register_depth, expected_depth)


def test_min_reg_depth_index():
    circuit = CircuitDAG(n_emitter=1, n_photon=2, n_classical=1)

    circuit.add(ops.Hadamard(register=0, reg_type="e"))
    circuit.add(ops.CNOT(control=0, control_type="e", target=0, target_type="p"))
    circuit.add(ops.CNOT(control=0, control_type="e", target=1, target_type="p"))

    circuit.add(ops.SigmaX(register=0, reg_type="p"))
    circuit.add(ops.Hadamard(register=0, reg_type="p"))

    expected_output = 1
    assert circuit.min_reg_depth_index("p") == expected_output


def test_sorted_reg_depth_index():
    circuit = CircuitDAG(n_emitter=1, n_photon=2, n_classical=1)

    circuit.add(ops.Hadamard(register=0, reg_type="e"))
    circuit.add(ops.CNOT(control=0, control_type="e", target=0, target_type="p"))
    circuit.add(ops.CNOT(control=0, control_type="e", target=1, target_type="p"))

    circuit.add(ops.SigmaX(register=0, reg_type="p"))
    circuit.add(ops.Hadamard(register=0, reg_type="p"))

    expected_output = np.array([1, 0])
    test.assert_equal(circuit.sorted_reg_depth_index("p"), expected_output)
