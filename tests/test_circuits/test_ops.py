import pytest

import src.ops as ops


def test_single_qubit_clifford_combo():
    clifford_iter = ops.one_qubit_cliffords()
    assert next(clifford_iter) == [ops.Identity, ops.Identity]
    assert next(clifford_iter) == [ops.Identity, ops.SigmaX]
    assert next(clifford_iter) == [ops.Identity, ops.SigmaY]
    assert next(clifford_iter) == [ops.Identity, ops.SigmaZ]

    assert next(clifford_iter) == [
        ops.Hadamard,
        ops.Phase,
        ops.Hadamard,
        ops.Phase,
        ops.Identity,
    ]
    assert next(clifford_iter) == [
        ops.Hadamard,
        ops.Phase,
        ops.Hadamard,
        ops.Phase,
        ops.SigmaX,
    ]
    assert next(clifford_iter) == [
        ops.Hadamard,
        ops.Phase,
        ops.Hadamard,
        ops.Phase,
        ops.SigmaY,
    ]
    assert next(clifford_iter) == [
        ops.Hadamard,
        ops.Phase,
        ops.Hadamard,
        ops.Phase,
        ops.SigmaZ,
    ]

    assert next(clifford_iter) == [ops.Hadamard, ops.Phase, ops.Identity]
    assert next(clifford_iter) == [ops.Hadamard, ops.Phase, ops.SigmaX]
    assert next(clifford_iter) == [ops.Hadamard, ops.Phase, ops.SigmaY]
    assert next(clifford_iter) == [ops.Hadamard, ops.Phase, ops.SigmaZ]

    assert next(clifford_iter) == [ops.Hadamard, ops.Identity]
    assert next(clifford_iter) == [ops.Hadamard, ops.SigmaX]
    assert next(clifford_iter) == [ops.Hadamard, ops.SigmaY]
    assert next(clifford_iter) == [ops.Hadamard, ops.SigmaZ]

    assert next(clifford_iter) == [ops.Phase, ops.Hadamard, ops.Phase, ops.Identity]
    assert next(clifford_iter) == [ops.Phase, ops.Hadamard, ops.Phase, ops.SigmaX]
    assert next(clifford_iter) == [ops.Phase, ops.Hadamard, ops.Phase, ops.SigmaY]
    assert next(clifford_iter) == [ops.Phase, ops.Hadamard, ops.Phase, ops.SigmaZ]

    assert next(clifford_iter) == [ops.Phase, ops.Identity]
    assert next(clifford_iter) == [ops.Phase, ops.SigmaX]
    assert next(clifford_iter) == [ops.Phase, ops.SigmaY]
    assert next(clifford_iter) == [ops.Phase, ops.SigmaZ]

    with pytest.raises(StopIteration):
        next(clifford_iter)


def test_unwrapping_base_gate_1():
    """Checks that an arbitrary, non-composed operation can be correctly unwrapped"""
    sigma_x_unwrapped = ops.SigmaX(register=1, reg_type="p").unwrap()
    assert len(sigma_x_unwrapped) == 1
    assert sigma_x_unwrapped[0].register == 1
    assert sigma_x_unwrapped[0].reg_type == "p"
    assert isinstance(sigma_x_unwrapped[0], ops.SigmaX)


def test_unwrapping_base_gate_2():
    """Checks that an arbitrary, non-composed operation can be correctly unwrapped"""
    sigma_cnot_unwrapped = ops.CNOT(
        control=1, control_type="e", target=2, target_type="p"
    ).unwrap()
    assert len(sigma_cnot_unwrapped) == 1
    assert sigma_cnot_unwrapped[0].control == 1
    assert sigma_cnot_unwrapped[0].control_type == "e"
    assert sigma_cnot_unwrapped[0].target == 2
    assert sigma_cnot_unwrapped[0].target_type == "p"
    assert isinstance(sigma_cnot_unwrapped[0], ops.CNOT)


def test_wrapper_gate_1():
    """Checks that an error is thrown in the wrapper is empty"""
    with pytest.raises(ValueError):
        ops.OneQubitGateWrapper([], register=0)


def test_wrapper_gate_2():
    """Checks that an error is thrown in the wrapper contains multi-qubit gates"""
    with pytest.raises(AssertionError):
        gates = [ops.CNOT, ops.Hadamard, ops.Phase]
        ops.OneQubitGateWrapper(gates)


def test_wrapper_gate_unwrap_1():
    """Test unwrap a single operation"""
    gates = [ops.Hadamard]
    operation = ops.OneQubitGateWrapper(gates)
    unwrapped = operation.unwrap()
    for i, op in enumerate(unwrapped):
        assert isinstance(op, gates[::-1][i])
        assert op.register == operation.register
        assert op.q_registers == operation.q_registers
        assert op.q_registers_type == operation.q_registers_type


def test_wrapper_gate_unwrap_2():
    """Test unwrap multiple operations"""
    gates = [ops.Hadamard, ops.Phase, ops.Hadamard, ops.Phase, ops.Identity]
    operation = ops.OneQubitGateWrapper(gates)
    unwrapped = operation.unwrap()
    for i, op in enumerate(unwrapped):
        assert isinstance(op, gates[::-1][i])
        assert op.register == operation.register
        assert op.q_registers == operation.q_registers
        assert op.q_registers_type == operation.q_registers_type
