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
Placement of `AdditionNoiseBase` noise around a controlled gate.

`CompilerBase.compile` decides, per leg of a controlled pair, whether that
leg's noise is applied before or after the gate, from its
`noise_parameters["After gate"]` flag. The two legs may disagree, and each
leg must land on its own side regardless of what the other one does.

These tests drive `compile()` end to end and compare against the same
circuit written out with explicit Pauli gates in the expected positions.
The probe noise is `PauliError`, which is a deterministic Clifford on every
backend (a coherent gate error, not a stochastic channel), so each expected
state is exact rather than statistical.
"""

import numpy as np
import pytest

import graphiq.circuit.ops as ops
import graphiq.noise.noise_models as nm
from graphiq.backends.density_matrix.compiler import DensityMatrixCompiler
from graphiq.backends.stabilizer.compiler import StabilizerCompiler
from graphiq.circuit.circuit_dag import CircuitDAG


def _before_gate(noise):
    """Flip an AdditionNoiseBase to pre-gate placement.

    No noise model exposes this through its constructor -- each builds its own
    `noise_parameters` dict and `AdditionNoiseBase.__init__` then stamps
    `"After gate": True` onto it -- so a mixed-position circuit can only be
    built by mutating the flag afterwards.
    """
    noise.noise_parameters["After gate"] = False
    return noise


def _noisy_cnot_circuit(control_noise, target_noise):
    """H(0), then a CNOT(0, 1) carrying one noise model per leg.

    Two things about the probe are load-bearing, and both were arrived at by
    watching a weaker one pass against the broken compiler:

    The Hadamard leaves qubit 0 in an X eigenstate, so the Pauli errors do not
    act as a global phase on a Z eigenstate -- which a density-matrix comparison
    could not see.

    The callers pair an X error on the *control* with a Z error on the
    *target*, because those are the two that anticommute across a CNOT. An X on
    the target, or a Z on the control, commutes with the gate, so moving that
    leg from one side to the other would leave the final state unchanged and
    the test could not tell a correct placement from a wrong one.
    """
    circuit = CircuitDAG(n_emitter=2, n_photon=0, n_classical=0)
    circuit.add(ops.Hadamard(register=0, reg_type="e", noise=nm.NoNoise()))
    cnot = ops.CNOT(
        control=0,
        control_type="e",
        target=1,
        target_type="e",
        noise=[control_noise, target_noise],
    )
    # `ControlledPairOperationBase.__init__` turns a *scalar* noise argument
    # into `2 * [noise]`, i.e. the same object in both slots -- mutating
    # "After gate" on that would flip both legs at once and silently test a
    # non-mixed case. Passing a list of two distinct instances avoids it.
    assert cnot.noise[0] is not cnot.noise[1]
    circuit.add(cnot)
    return circuit, cnot


def _explicit_circuit(pre_gates, post_gates):
    """The same H(0) and CNOT(0, 1), with explicit noiseless Pauli gates.

    `pre_gates`/`post_gates` are (operation class, register) pairs placed
    before and after the CNOT.
    """
    circuit = CircuitDAG(n_emitter=2, n_photon=0, n_classical=0)
    circuit.add(ops.Hadamard(register=0, reg_type="e", noise=nm.NoNoise()))
    for gate, register in pre_gates:
        circuit.add(gate(register=register, reg_type="e", noise=nm.NoNoise()))
    circuit.add(
        ops.CNOT(
            control=0,
            control_type="e",
            target=1,
            target_type="e",
            noise=[nm.NoNoise(), nm.NoNoise()],
        )
    )
    for gate, register in post_gates:
        circuit.add(gate(register=register, reg_type="e", noise=nm.NoNoise()))
    return circuit


def _compile_dm(circuit):
    compiler = DensityMatrixCompiler()
    compiler.noise_simulation = True
    return compiler.compile(circuit).rep_data.data



# --------------------------------------------------------------------------
# The two positions that agree. No coverage existed for either before this
# file, so these are written to pass against the unmodified compiler and
# then to pin that behaviour across any restructuring of the placement code.
# --------------------------------------------------------------------------


def test_both_legs_after_gate():
    circuit, _ = _noisy_cnot_circuit(nm.PauliError("X"), nm.PauliError("Z"))
    expected = _explicit_circuit([], [(ops.SigmaX, 0), (ops.SigmaZ, 1)])
    assert np.allclose(_compile_dm(circuit), _compile_dm(expected))


def test_neither_leg_after_gate():
    circuit, _ = _noisy_cnot_circuit(
        _before_gate(nm.PauliError("X")), _before_gate(nm.PauliError("Z"))
    )
    expected = _explicit_circuit([(ops.SigmaX, 0), (ops.SigmaZ, 1)], [])
    assert np.allclose(_compile_dm(circuit), _compile_dm(expected))


# --------------------------------------------------------------------------
# The two positions that disagree.
# --------------------------------------------------------------------------


def test_control_after_target_before():
    circuit, _ = _noisy_cnot_circuit(
        nm.PauliError("X"), _before_gate(nm.PauliError("Z"))
    )
    expected = _explicit_circuit([(ops.SigmaZ, 1)], [(ops.SigmaX, 0)])
    assert np.allclose(_compile_dm(circuit), _compile_dm(expected))


def test_control_before_target_after():
    circuit, _ = _noisy_cnot_circuit(
        _before_gate(nm.PauliError("X")), nm.PauliError("Z")
    )
    expected = _explicit_circuit([(ops.SigmaX, 0)], [(ops.SigmaZ, 1)])
    assert np.allclose(_compile_dm(circuit), _compile_dm(expected))


def test_mixed_position_differs_from_both_after():
    """A mixed placement must not coincide with the all-after placement.

    Guards the two tests above against passing for the wrong reason: if the
    chosen Paulis commuted through the CNOT, moving one leg across the gate
    would be unobservable and a dropped leg could look correct.
    """
    mixed, _ = _noisy_cnot_circuit(nm.PauliError("X"), _before_gate(nm.PauliError("Z")))
    both_after, _ = _noisy_cnot_circuit(nm.PauliError("X"), nm.PauliError("Z"))
    assert not np.allclose(_compile_dm(mixed), _compile_dm(both_after))



# --------------------------------------------------------------------------
# `compile()` rebinds `op.noise` while placing the legs. It must put the
# caller's own list back, on every exit path -- the circuit belongs to the
# caller and is routinely compiled more than once.
# --------------------------------------------------------------------------


def test_op_noise_restored_after_compile():
    circuit, cnot = _noisy_cnot_circuit(
        _before_gate(nm.PauliError("X")), nm.PauliError("Z")
    )
    original = cnot.noise
    original_legs = list(cnot.noise)
    _compile_dm(circuit)
    assert cnot.noise is original
    assert list(cnot.noise) == original_legs


def test_op_noise_restored_when_gate_compilation_raises(monkeypatch):
    """A failed compile must not leave the circuit poisoned.

    Without an unwind, `op.noise` keeps a half-built list holding the
    `nm.NoNoise` *class* in one slot. That slot then fails the
    `isinstance(..., AdditionNoiseBase)` guard on the next compile, so a
    circuit with nothing wrong with it reports a defect it does not have.
    """
    circuit, cnot = _noisy_cnot_circuit(
        _before_gate(nm.PauliError("X")), nm.PauliError("Z")
    )
    original = cnot.noise
    original_legs = list(cnot.noise)

    def _boom(self, state, op, n_quantum, q_index, classical_registers):
        raise RuntimeError("gate compilation failed")

    monkeypatch.setattr(DensityMatrixCompiler, "compile_one_gate", _boom)
    with pytest.raises(RuntimeError, match="gate compilation failed"):
        _compile_dm(circuit)

    assert cnot.noise is original
    assert list(cnot.noise) == original_legs

    # The real symptom: the next compile of this circuit must still work.
    monkeypatch.undo()
    expected = _explicit_circuit([(ops.SigmaX, 0)], [(ops.SigmaZ, 1)])
    assert np.allclose(_compile_dm(circuit), _compile_dm(expected))


@pytest.mark.parametrize("failing_call", [1, 2])
def test_op_noise_restored_when_noise_application_raises(monkeypatch, failing_call):
    """A raise while a half of the split noise is bound must restore the gate's noise.

    The CNOT's noise is split into a pre-gate and a post-gate half, and `op.noise` is
    bound to each in turn. Call 1 is the pre-gate half, call 2 the post-gate half; the
    other ops carry no noise and never reach `_apply_additional_noise`.
    """
    circuit, cnot = _noisy_cnot_circuit(
        _before_gate(nm.PauliError("X")), nm.PauliError("Z")
    )
    original = cnot.noise
    original_legs = list(cnot.noise)
    apply_noise = DensityMatrixCompiler._apply_additional_noise
    bound = []

    def _fail_on_call(self, state, op, n_quantum, q_index):
        assert op is cnot
        bound.append([not isinstance(leg, nm.NoNoise) for leg in op.noise])
        if len(bound) == failing_call:
            raise RuntimeError("noise application failed")
        return apply_noise(self, state, op, n_quantum, q_index)

    monkeypatch.setattr(DensityMatrixCompiler, "_apply_additional_noise", _fail_on_call)
    with pytest.raises(RuntimeError, match="noise application failed"):
        _compile_dm(circuit)

    # Which leg carries real noise in each bound half: the X leg before, the Z leg after.
    assert bound == [[True, False], [False, True]][:failing_call]
    assert cnot.noise is original
    assert list(cnot.noise) == original_legs

    monkeypatch.undo()
    expected = _explicit_circuit([(ops.SigmaX, 0)], [(ops.SigmaZ, 1)])
    assert np.allclose(_compile_dm(circuit), _compile_dm(expected))


def test_non_addition_noise_on_a_leg_is_rejected():
    """Both legs must be `AdditionNoiseBase`; differing *positions* are fine."""
    replacement = nm.OneQubitGateReplacement(np.eye(2))
    circuit, _ = _noisy_cnot_circuit(replacement, nm.PauliError("X"))
    with pytest.raises(ValueError, match="AdditionNoiseBase"):
        _compile_dm(circuit)


@pytest.mark.parametrize("compiler_class", [DensityMatrixCompiler, StabilizerCompiler])
def test_replacement_noise_on_measurement_z_is_refused(compiler_class):
    """No measurement error model is implemented for replacement noise, so the
    compiler must refuse it rather than drop the noise (or the measurement)."""
    circuit = CircuitDAG(n_emitter=1, n_photon=0, n_classical=1)
    circuit.add(ops.Hadamard(register=0, reg_type="e", noise=nm.NoNoise()))
    circuit.add(
        ops.MeasurementZ(
            register=0,
            reg_type="e",
            c_register=0,
            noise=nm.OneQubitGateReplacement(np.eye(2)),
        )
    )
    compiler = compiler_class()
    compiler.noise_simulation = True
    with pytest.raises(NotImplementedError, match="MeasurementZ"):
        compiler.compile(circuit)
