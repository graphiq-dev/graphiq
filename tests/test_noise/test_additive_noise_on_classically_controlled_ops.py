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
"""
Additive noise on a classically controlled op's legs, on both backends.

`ClassicalCNOT`/`ClassicalCZ`/`MeasurementCNOTandReset` (a
`ClassicalControlledPairOperationBase`) each carry one noise model per
register. `CompilerBase.compile` already splits those legs into a
pre-gate and a post-gate half by each leg's `"After gate"` flag whenever every
leg is an `AdditionNoiseBase`, and calls `_apply_additional_noise` on each
half -- this file exercises the backend arm that has to honour that split for
this op family, on both the stabilizer and the density-matrix backend.

For `MeasurementCNOTandReset`, "the gate" runs as: measure the control,
conditionally apply the correction to the target, then reset the control. A
pre-gate leg on the control therefore acts on the qubit *before* it is
measured (so it can change the outcome), and a post-gate leg on the control
acts *after* the reset -- the only reading consistent with where
`compile_one_gate` measures and resets it. Every circuit below starts from
the all-zero state and every measurement outcome is deterministic once the
noise has acted, so `measurement_determinism = 0` never has to break a tie.

Each behavioural test compiles the noisy circuit and an equivalent
circuit written out with an explicit, noiseless Pauli gate in the position
the noise should land, and asserts the two compile to the same state --
mirroring `test_noise_placement.py`'s comparison style. A stabilizer compile
is converted to a density matrix via `QuantumState.convert_representation`
before the comparison, since both backends are being driven through the same
assertion.
"""

import numpy as np
import pytest

import graphiq.circuit.ops as ops
import graphiq.noise.noise_models as nm
from graphiq.backends.density_matrix.compiler import DensityMatrixCompiler
from graphiq.backends.stabilizer.compiler import StabilizerCompiler
from graphiq.circuit.circuit_dag import CircuitDAG


def _pauli(letter, after_gate):
    """A `PauliError` leg with its `"After gate"` placement spelled out.

    `PauliError` takes no placement argument -- `AdditionNoiseBase.__init__`
    always stamps `noise_parameters["After gate"] = True` on construction --
    so a pre-gate leg is only reachable by flipping the flag afterwards.
    """
    noise = nm.PauliError(letter)
    noise.noise_parameters["After gate"] = after_gate
    return noise


def _cnot_and_reset_circuit(noise_legs):
    """`MeasurementCNOTandReset(control=0/e, target=0/p)` on `|0>|0>`."""
    circuit = CircuitDAG(n_emitter=1, n_photon=1, n_classical=1)
    circuit.add(
        ops.MeasurementCNOTandReset(
            control=0,
            control_type="e",
            target=0,
            target_type="p",
            c_register=0,
            noise=list(noise_legs),
        )
    )
    return circuit


def _cnot_and_reset_explicit(pre_control=False, post_control=False):
    """The same op, noiseless, with an explicit `SigmaX(control)` where a
    noisy leg should land."""
    circuit = CircuitDAG(n_emitter=1, n_photon=1, n_classical=1)
    if pre_control:
        circuit.add(ops.SigmaX(register=0, reg_type="e", noise=nm.NoNoise()))
    circuit.add(
        ops.MeasurementCNOTandReset(
            control=0,
            control_type="e",
            target=0,
            target_type="p",
            c_register=0,
            noise=[nm.NoNoise(), nm.NoNoise()],
        )
    )
    if post_control:
        circuit.add(ops.SigmaX(register=0, reg_type="e", noise=nm.NoNoise()))
    return circuit






def _compile(compiler_class, circuit):
    compiler = compiler_class()
    compiler.noise_simulation = True
    compiler.measurement_determinism = 0
    state = compiler.compile(circuit)
    if compiler_class is StabilizerCompiler:
        state.convert_representation("dm")
    return state.rep_data.data


@pytest.mark.parametrize("compiler_class", [StabilizerCompiler, DensityMatrixCompiler])
def test_pre_gate_noise_on_the_control_flips_the_branch(compiler_class):
    noisy = _cnot_and_reset_circuit([_pauli("X", after_gate=False), nm.NoNoise()])
    expected = _cnot_and_reset_explicit(pre_control=True)
    assert np.allclose(
        _compile(compiler_class, noisy), _compile(compiler_class, expected)
    )

    # Without the noise, the same circuit measures 0 and touches nothing.
    plain = _cnot_and_reset_explicit()
    assert not np.allclose(
        _compile(compiler_class, noisy), _compile(compiler_class, plain)
    )


@pytest.mark.parametrize("compiler_class", [StabilizerCompiler, DensityMatrixCompiler])
def test_post_gate_noise_on_the_control_lands_after_the_reset(compiler_class):
    noisy = _cnot_and_reset_circuit([_pauli("X", after_gate=True), nm.NoNoise()])
    expected = _cnot_and_reset_explicit(post_control=True)
    assert np.allclose(
        _compile(compiler_class, noisy), _compile(compiler_class, expected)
    )

    plain = _cnot_and_reset_explicit()
    assert not np.allclose(
        _compile(compiler_class, noisy), _compile(compiler_class, plain)
    )




def test_the_two_backends_agree():
    noisy_circuits = [
        _cnot_and_reset_circuit([_pauli("X", after_gate=False), nm.NoNoise()]),
        _cnot_and_reset_circuit([_pauli("X", after_gate=True), nm.NoNoise()]),
    ]
    for circuit in noisy_circuits:
        stabilizer_result = _compile(StabilizerCompiler, circuit)
        dm_result = _compile(DensityMatrixCompiler, circuit)
        assert np.allclose(stabilizer_result, dm_result)


def test_an_op_outside_every_arm_still_raises_on_the_stabilizer_backend():
    # The dispatcher never routes a MeasurementZ through
    # `_apply_additional_noise` (it is not a controlled op), so this is only
    # reachable by calling the method directly -- which is exactly what pins
    # that the bare `pass` this arm used to be is gone.
    op = ops.MeasurementZ(
        register=0, reg_type="e", c_register=0, noise=nm.PauliError("X")
    )
    with pytest.raises(ValueError):
        StabilizerCompiler()._apply_additional_noise(
            None, op, 1, lambda reg, reg_type: 0
        )
