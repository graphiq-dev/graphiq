# Copyright (c) 2022-2024 Quantum Bridge Technologies Inc.
# Copyright (c) 2022-2024 Ki3 Photonics Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import numpy as np
import pytest

import graphiq.backends.density_matrix.functions as dmf
import graphiq.circuit.ops as ops


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


def test_get_local_clifford_matrix_by_name():
    assert np.allclose(ops.local_clifford_to_matrix_map(ops.Hadamard), dmf.hadamard())
    result = ops.local_clifford_to_matrix_map([ops.Hadamard, ops.Phase, ops.SigmaX])
    expected = dmf.hadamard() @ dmf.phase() @ dmf.sigmax()
    assert np.allclose(result, expected)
    assert ops.find_local_clifford_by_matrix(expected) == [
        ops.Hadamard,
        ops.Phase,
        ops.SigmaX,
    ]
    expected2 = dmf.hadamard() @ dmf.phase() @ dmf.phase() @ dmf.hadamard()
    assert ops.find_local_clifford_by_matrix(expected2) == [ops.Identity, ops.SigmaX]
    expected3 = dmf.phase() @ dmf.phase() @ dmf.hadamard() @ dmf.hadamard()
    assert ops.find_local_clifford_by_matrix(expected3) == [ops.Identity, ops.SigmaZ]
    assert ops.simplify_local_clifford(
        [ops.Phase, ops.Phase, ops.Hadamard, ops.Hadamard]
    ) == [ops.Identity, ops.SigmaZ]


def test_op():
    u1 = dmf.phase() @ dmf.hadamard()
    u2 = dmf.hadamard() @ dmf.phase() @ dmf.hadamard() @ dmf.phase() @ dmf.sigmax()
    print(f"is u1 equivalent to u2 = {dmf.check_equivalent_unitaries(u1, u2)}")
    u3 = dmf.hadamard() @ dmf.phase() @ dmf.hadamard() @ dmf.phase()
    u4 = dmf.phase() @ dmf.hadamard() @ dmf.sigmax()
    print(f"is u3 equivalent to u4 = {dmf.check_equivalent_unitaries(u3, u4)}")
