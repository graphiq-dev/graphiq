import pytest
import random

from src.circuit import CircuitDAG
import src.ops as ops
from tests.test_flags import visualization


# Test initialization
# Test registers depth after circuit initialization
# Better with @pytest.mark.parametrize for multiple cases check
def test_initialization():
    circuit = CircuitDAG(n_emitter=1, n_photon=4, n_classical=1)

    assert circuit.register_depth == {"e": [0], "p": [0, 0, 0, 0], "c": [0]}


# Test add operation
# Test registers depth after add new operation to registers
def test_add_operation():
    circuit = CircuitDAG(n_emitter=1, n_photon=1, n_classical=1)
    expected_depth = {"e": [0], "p": [0], "c": [0]}

    circuit.add(ops.SigmaX(register=0, reg_type="p"))
    expected_depth["p"][0] += 1

    assert circuit.register_depth == expected_depth


# Test insert operation at
# Test registers depth after insert new operation at nodes
def test_insert_operation_at():
    circuit = CircuitDAG(n_emitter=1, n_photon=1, n_classical=1)
    expected_depth = {"e": [0], "p": [0], "c": [0]}

    circuit.add(ops.SigmaX(register=0, reg_type="p"))
    expected_depth["p"][0] += 1

    circuit.insert_at(operation=ops.SigmaY(register=0, reg_type="p"), edges=[("p0_in", 1, "p0")])
    expected_depth["p"][0] += 1

    assert circuit.register_depth == expected_depth

# Test replace operation
# Test registers depth after replace operation at nodes
def test_replace_operation():
    circuit = CircuitDAG(n_emitter=1, n_photon=1, n_classical=1)
    expected_depth = {"e": [0], "p": [0], "c": [0]}

    circuit.add(ops.SigmaX(register=0, reg_type="p"))
    expected_depth["p"][0] += 1

    circuit.replace_op(node=1, new_operation=ops.SigmaY(register=0, reg_type="p"))

    assert circuit.register_depth == expected_depth


# Test remove operation
# Test registers depth after remove operation at nodes
def test_remove_operation():
    circuit = CircuitDAG(n_emitter=1, n_photon=1, n_classical=1)
    expected_depth = {"e": [0], "p": [0], "c": [0]}

    circuit.add(ops.SigmaX(register=0, reg_type="p"))
    expected_depth["p"][0] += 1

    circuit.remove_op(node=1)
    expected_depth["p"][0] -= 1

    assert circuit.register_depth == expected_depth

