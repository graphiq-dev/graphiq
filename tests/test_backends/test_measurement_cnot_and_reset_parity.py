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
Parity between the density-matrix and stabilizer backends for
`MeasurementCNOTandReset`, and the related classically-controlled family
(`ClassicalCNOT`, `ClassicalCZ`) on `MixedStabilizer`.

Every measurement-outcome assertion here pins `measurement_determinism = 1`.
At `determinism = 0` the emitter is projected to |0> anyway and the recorded
outcome is 0.0, so a missing reset or a missing register write would go
unseen. The default is "probabilistic", so a test that does not pin
determinism would catch either only about half the time.
"""

import numpy as np
import pytest

import graphiq.backends.density_matrix.functions as dmf
import graphiq.backends.state_rep_conversion as rc
import graphiq.circuit.ops as ops
import graphiq.noise.noise_models as nm
from graphiq.backends.density_matrix.compiler import DensityMatrixCompiler
from graphiq.backends.stabilizer.compiler import StabilizerCompiler
from graphiq.circuit.circuit_dag import CircuitDAG

# n_photon=1, n_emitter=1: with ordering photons-then-emitters
# (CompilerBase.reg_to_index_func), the emitter sits at qubit index 1.
_EMITTER_INDEX = 1


def _h_cnot_reset_circuit(n_classical=1):
    """H(e0), CNOT(e0->p0), MeasurementCNOTandReset(e0->p0, c_register=0).

    The Hadamard leaves the control in an X eigenstate, so a Z-basis
    measurement has p(1) != 0, and `measurement_determinism = 1`'s "default to
    1 unless p(1) is 0" rule forces the outcome to 1 -- the branch where a
    missing reset is observable.
    """
    circuit = CircuitDAG(n_emitter=1, n_photon=1, n_classical=n_classical)
    circuit.add(ops.Hadamard(register=0, reg_type="e"))
    circuit.add(ops.CNOT(control=0, control_type="e", target=0, target_type="p"))
    circuit.add(
        ops.MeasurementCNOTandReset(
            control=0, control_type="e", target=0, target_type="p", c_register=0
        )
    )
    return circuit


def _record_classical_registers(compiler):
    """
    Make ``compiler.classical_registers`` refer to the classical-register array
    that ``compile`` allocates and passes to every ``compile_one_gate`` call.
    """
    compile_one_gate = compiler.compile_one_gate

    def recording(state, op, n_quantum, q_index, classical_registers):
        compiler.classical_registers = classical_registers
        return compile_one_gate(state, op, n_quantum, q_index, classical_registers)

    compiler.compile_one_gate = recording
    return compiler


def _compile_dm(circuit):
    compiler = DensityMatrixCompiler()
    compiler.measurement_determinism = 1
    state = _record_classical_registers(compiler).compile(circuit)
    return state, compiler


def _compile_stabilizer(circuit):
    compiler = StabilizerCompiler()
    compiler.measurement_determinism = 1
    state = _record_classical_registers(compiler).compile(circuit)
    return state, compiler


def _stabilizer_qubit_density(state, qubit, n_qubits):
    """The reduced density matrix of one qubit of a (non-mixed) Stabilizer.

    Converts the whole tableau to a density matrix via
    `state_rep_conversion.stabilizer_to_density` -- the same utility
    `test_stabilizer_compiler.py` uses -- and traces the rest away with
    `dmf.partial_trace`, so the index arithmetic is never hand-rolled.
    """
    tableau = state.rep_data.tableau.to_stabilizer()
    rho = rc.stabilizer_to_density(tableau)
    return dmf.partial_trace(rho, keep=[qubit], dims=n_qubits * [2])


# ---------------------------------------------------------------------------
# The emitter is |0> after MeasurementCNOTandReset, on both backends, pinned
# at determinism = 1.
# ---------------------------------------------------------------------------


def test_dm_emitter_is_reset_after_measurement_cnot_and_reset():
    circuit = _h_cnot_reset_circuit()
    state, _ = _compile_dm(circuit)
    reduced = dmf.partial_trace(state.rep_data.data, keep=[_EMITTER_INDEX], dims=[2, 2])
    assert np.allclose(reduced, [[1.0, 0.0], [0.0, 0.0]])


def test_stabilizer_emitter_is_reset_after_measurement_cnot_and_reset():
    circuit = _h_cnot_reset_circuit()
    state, _ = _compile_stabilizer(circuit)
    reduced = _stabilizer_qubit_density(state, _EMITTER_INDEX, n_qubits=2)
    assert np.allclose(reduced, [[1.0, 0.0], [0.0, 0.0]])


# ---------------------------------------------------------------------------
# classical_registers[c_register] holds the outcome on both backends, pinned
# at determinism = 1, where the outcome is 1.0.
# ---------------------------------------------------------------------------


def test_dm_creg_records_the_outcome():
    circuit = _h_cnot_reset_circuit()
    _, compiler = _compile_dm(circuit)
    assert compiler.classical_registers[0] == 1.0


def test_stabilizer_creg_records_the_outcome():
    circuit = _h_cnot_reset_circuit()
    _, compiler = _compile_stabilizer(circuit)
    assert compiler.classical_registers[0] == 1.0


# ---------------------------------------------------------------------------
# A circuit that re-uses the emitter after the reset, as feed-forward work
# that reads the emitter after a measurement does. The tests above check the
# state right after the op; these check what a later op sees.
#
# A second op (MeasurementZ, not another MeasurementCNOTandReset) reads the
# emitter directly, so this depends only on the reset, not on
# MeasurementCNOTandReset's own register write.
# ---------------------------------------------------------------------------


def _reuse_circuit():
    circuit = _h_cnot_reset_circuit(n_classical=2)
    circuit.add(ops.MeasurementZ(register=0, reg_type="e", c_register=1))
    return circuit


def test_dm_emitter_reuse_after_reset():
    circuit = _reuse_circuit()
    _, compiler = _compile_dm(circuit)
    # First outcome: the H'd control gives p(1) != 0, forced to 1.
    assert compiler.classical_registers[0] == 1.0
    # Second outcome: if the control was actually reset to |0>, p(1) == 0, so
    # determinism = 1's "default to 1 unless p(1) is 0" rule forces 0. A
    # not-reset emitter (still |1> from the first measurement) would force 1.
    assert compiler.classical_registers[1] == 0.0


def test_stabilizer_emitter_reuse_after_reset():
    circuit = _reuse_circuit()
    _, compiler = _compile_stabilizer(circuit)
    assert compiler.classical_registers[0] == 1.0
    assert compiler.classical_registers[1] == 0.0


# ---------------------------------------------------------------------------
# MixedStabilizer writes a scalar, not a list, for MeasurementCNOTandReset,
# ClassicalCNOT and ClassicalCZ. The outcome has one entry per mixture
# branch, and a classical_registers slot holds one value, so writing the
# whole list would raise ValueError.
#
# `noise_simulation = True` with every op carrying NoNoise still routes
# through compile_one_gate (CompilerBase.compile's no_noise branch), but now
# against a MixedStabilizer state -- the same construction
# test_noise_placement.py already relies on for this backend.
# ---------------------------------------------------------------------------


def _mixed_stabilizer_circuit(op_cls):
    """SigmaX(e0) forces the control to |1> outcome deterministically, then
    op_cls(e0->p0) under noise_simulation=True so the compiler builds a
    MixedStabilizer.
    """
    circuit = CircuitDAG(n_emitter=1, n_photon=1, n_classical=1)
    circuit.add(ops.SigmaX(register=0, reg_type="e", noise=nm.NoNoise()))
    circuit.add(
        op_cls(
            control=0,
            control_type="e",
            target=0,
            target_type="p",
            c_register=0,
            noise=nm.NoNoise(),
        )
    )
    return circuit


def _compile_mixed_stabilizer(circuit):
    compiler = StabilizerCompiler()
    compiler.noise_simulation = True
    compiler.measurement_determinism = 1
    state = _record_classical_registers(compiler).compile(circuit)
    return state, compiler


@pytest.mark.parametrize(
    "op_cls", [ops.MeasurementCNOTandReset, ops.ClassicalCNOT, ops.ClassicalCZ]
)
def test_mixed_stabilizer_writes_a_scalar_not_a_list(op_cls):
    circuit = _mixed_stabilizer_circuit(op_cls)
    _, compiler = _compile_mixed_stabilizer(circuit)

    value = compiler.classical_registers[0]
    assert not isinstance(value, (list, np.ndarray))
    assert value == 1.0


# ---------------------------------------------------------------------------
# No stray stdout from a MixedStabilizer ClassicalCNOT compile.
# ---------------------------------------------------------------------------


def test_mixed_stabilizer_classical_cnot_compiles_silently(capsys):
    circuit = _mixed_stabilizer_circuit(ops.ClassicalCNOT)
    _compile_mixed_stabilizer(circuit)
    captured = capsys.readouterr()
    assert captured.out == ""
