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

import graphiq.benchmarks.circuits as circ
import graphiq.circuit.ops as ops
import graphiq.noise.noise_models as nm
from graphiq.backends.density_matrix.compiler import DensityMatrixCompiler
from graphiq.backends.stabilizer.compiler import StabilizerCompiler
from graphiq.benchmarks.alternate_circuits import (
    noise_model_loss_and_depolarizing,
    noise_model_pure_loss,
)
from graphiq.circuit.circuit_dag import CircuitDAG
from graphiq.noise.monte_carlo_noise import McNoiseMap, MonteCarloNoise
from graphiq.solvers.solver_base import SolverBase


def _assert_every_leg_is_no_noise(op):
    if isinstance(op.noise, list):
        assert all(isinstance(leg, nm.NoNoise) for leg in op.noise)
    else:
        assert isinstance(op.noise, nm.NoNoise)


def test_assign_noise_leaves_the_source_circuit_alone():
    hadamard_noise = nm.DepolarizingNoise(0.1)
    wrapper_noise = nm.DepolarizingNoise(0.2)
    pair_noise = nm.DepolarizingNoise(0.3)

    circuit = CircuitDAG(n_emitter=1, n_photon=2, n_classical=1)
    circuit.add(ops.Hadamard(register=0, reg_type="e"))
    circuit.add(
        ops.OneQubitGateWrapper([ops.Phase, ops.Identity], register=0, reg_type="e")
    )
    circuit.add(ops.CNOT(control=0, control_type="e", target=0, target_type="p"))
    circuit.add(
        ops.MeasurementCNOTandReset(
            control=0, control_type="e", target=1, target_type="p", c_register=0
        )
    )

    mapping = {
        "e": {
            "Hadamard": hadamard_noise,
            "Phase": wrapper_noise,
            "Identity": wrapper_noise,
        },
        "p": {},
        "ee": {},
        "ep": {"CNOT": pair_noise, "MeasurementCNOTandReset": pair_noise},
    }

    circuit.assign_noise(mapping)

    for op in circuit.sequence():
        if isinstance(op, (ops.Input, ops.Output)):
            continue
        _assert_every_leg_is_no_noise(op)


def test_assign_noise_shares_no_op_with_the_source():
    circuit = CircuitDAG(n_emitter=1, n_photon=1, n_classical=0)
    circuit.add(ops.Hadamard(register=0, reg_type="e"))
    circuit.add(ops.CNOT(control=0, control_type="e", target=0, target_type="p"))

    mapping = {
        "e": {"Hadamard": nm.DepolarizingNoise(0.1)},
        "p": {},
        "ee": {},
        "ep": {"CNOT": nm.DepolarizingNoise(0.2)},
    }

    noisy = circuit.assign_noise(mapping)

    source_ids = {
        id(op)
        for op in circuit.sequence()
        if not isinstance(op, (ops.Input, ops.Output))
    }
    result_ids = {
        id(op) for op in noisy.sequence() if not isinstance(op, (ops.Input, ops.Output))
    }
    assert source_ids.isdisjoint(result_ids)


def test_assign_noise_twice_with_different_mappings_is_independent():
    noise_a = nm.DepolarizingNoise(0.1)
    noise_b = nm.DepolarizingNoise(0.9)

    circuit = CircuitDAG(n_emitter=1, n_photon=0, n_classical=0)
    circuit.add(ops.Hadamard(register=0, reg_type="e"))

    mapping_a = {"e": {"Hadamard": noise_a}, "p": {}, "ee": {}, "ep": {}}
    mapping_b = {"e": {"Hadamard": noise_b}, "p": {}, "ee": {}, "ep": {}}

    result_a = circuit.assign_noise(mapping_a)
    circuit.assign_noise(mapping_b)

    hadamards_a = [op for op in result_a.sequence() if isinstance(op, ops.Hadamard)]
    assert len(hadamards_a) == 1
    assert hadamards_a[0].noise is noise_a


def test_monte_carlo_trials_leave_the_source_circuit_alone():
    circuit, _ = circ.ghz4_state_circuit()

    mc_noise_model = McNoiseMap()
    mc_noise_model.add_gate_noise("e", "Hadamard", [(nm.PauliError("X"), 1.0)])

    mc = MonteCarloNoise(circuit, mc_noise_model=mc_noise_model, seed=7)
    mc.one_run()
    mc.one_run()

    for op in circuit.sequence():
        if isinstance(op, (ops.Input, ops.Output)):
            continue
        if isinstance(op.noise, list):
            assert all(isinstance(leg, nm.NoNoise) for leg in op.noise)
        else:
            assert isinstance(op.noise, nm.NoNoise)


def _one_emitter_wrapper_circuit(operations, noise=nm.NoNoise()):
    circuit = CircuitDAG(n_emitter=1, n_photon=0, n_classical=0)
    circuit.add(
        ops.OneQubitGateWrapper(operations, register=0, reg_type="e", noise=noise)
    )
    return circuit


def _emitter_only_mapping(emitter_section):
    return {"e": emitter_section, "p": {}, "ee": {}, "ep": {}}


def _the_wrapper(circuit):
    wrappers = [
        op for op in circuit.sequence() if isinstance(op, ops.OneQubitGateWrapper)
    ]
    assert len(wrappers) == 1
    return wrappers[0]


def _noisy_state(circuit):
    compiler = DensityMatrixCompiler()
    compiler.noise_simulation = True
    return compiler.compile(circuit=circuit).rep_data.data


def test_assign_noise_pairs_wrapper_noise_with_its_sub_gate():
    # a wrapper pairs noise[i] with operations[i], so the model mapped to Hadamard must end up
    # on the Hadamard sub-gate, not on the sub-gate at the mirrored position in unwrap() order
    hadamard_noise = nm.DepolarizingNoise(0.1)
    circuit = _one_emitter_wrapper_circuit([ops.Hadamard, ops.Phase])

    noisy = circuit.assign_noise(_emitter_only_mapping({"Hadamard": hadamard_noise}))

    noise_by_gate = {
        type(gate).__name__: gate.noise for gate in _the_wrapper(noisy).unwrap()
    }
    assert noise_by_gate["Hadamard"] is hadamard_noise
    assert isinstance(noise_by_gate["Phase"], nm.NoNoise)


def test_assign_noise_wrapper_noise_reaches_the_right_sub_gate_in_the_compiled_state():
    # [Hadamard, SigmaX] applies SigmaX first, then Hadamard; a Z error on the Hadamard gives a
    # different state from a Z error on the SigmaX, so the compiled state shows which gate got it
    circuit = _one_emitter_wrapper_circuit([ops.Hadamard, ops.SigmaX])
    noisy = circuit.assign_noise(
        _emitter_only_mapping({"Hadamard": nm.PauliError("Z")})
    )

    reference = CircuitDAG(n_emitter=1, n_photon=0, n_classical=0)
    reference.add(ops.SigmaX(register=0, reg_type="e"))
    reference.add(ops.Hadamard(register=0, reg_type="e", noise=nm.PauliError("Z")))
    control = CircuitDAG(n_emitter=1, n_photon=0, n_classical=0)
    control.add(ops.SigmaX(register=0, reg_type="e", noise=nm.PauliError("Z")))
    control.add(ops.Hadamard(register=0, reg_type="e"))

    state = _noisy_state(noisy)
    reference_state = _noisy_state(reference)
    control_state = _noisy_state(control)

    assert not np.allclose(reference_state, control_state)
    assert np.allclose(state, reference_state)
    assert not np.allclose(state, control_state)


def test_assign_noise_leaves_a_whole_wrapper_model_in_place():
    # a wrapper carrying one model for the whole wrapper keeps it; the per-gate entries of the
    # mapping are not applied to it
    wrapper_noise = nm.PauliError("X")
    circuit = _one_emitter_wrapper_circuit(
        [ops.Hadamard, ops.SigmaX], noise=wrapper_noise
    )

    noisy = circuit.assign_noise(
        _emitter_only_mapping({"Hadamard": nm.DepolarizingNoise(0.1)})
    )

    wrapper = _the_wrapper(noisy)
    assert isinstance(wrapper.noise, nm.PauliError)
    assert wrapper.noise.noise_parameters == wrapper_noise.noise_parameters
    gates = wrapper.unwrap()
    assert len(gates) == 3
    assert {type(gate).__name__ for gate in gates} == {
        "Hadamard",
        "SigmaX",
        "Identity",
    }


class _MinimalSolver(SolverBase):
    """A concrete SolverBase subclass with no search behaviour, so that ``_identify_noise`` can be called
    without constructing a full solver."""

    def solve(self, *args):
        raise NotImplementedError


def test_assign_noise_honours_per_leg_keys():
    control_noise = nm.DepolarizingNoise(0.1)
    circuit = CircuitDAG(n_emitter=2, n_photon=0, n_classical=0)
    circuit.add(ops.CNOT(control=0, control_type="e", target=1, target_type="e"))

    mapping = {
        "e": {},
        "p": {},
        "ee": {"CNOT_control": control_noise},
        "ep": {},
    }
    noisy = circuit.assign_noise(mapping)

    cnots = [op for op in noisy.sequence() if isinstance(op, ops.CNOT)]
    assert len(cnots) == 1
    assert cnots[0].noise[0] is control_noise
    assert isinstance(cnots[0].noise[1], nm.NoNoise)


def test_assign_noise_and_identify_noise_agree():
    ee_control_noise = nm.DepolarizingNoise(0.1)
    ee_target_noise = nm.DepolarizingNoise(0.2)
    ep_noise = nm.DepolarizingNoise(0.3)
    hadamard_noise = nm.DepolarizingNoise(0.4)

    mapping = {
        "e": {"Hadamard": hadamard_noise},
        "p": {},
        "ee": {
            "CNOT_control": ee_control_noise,
            "CNOT_target": ee_target_noise,
        },
        "ep": {"MeasurementCNOTandReset": ep_noise},
    }

    circuit = CircuitDAG(n_emitter=2, n_photon=1, n_classical=1)
    circuit.add(ops.Hadamard(register=0, reg_type="e"))
    circuit.add(ops.CNOT(control=0, control_type="e", target=1, target_type="e"))
    circuit.add(
        ops.MeasurementCNOTandReset(
            control=0, control_type="e", target=0, target_type="p", c_register=0
        )
    )

    noisy = circuit.assign_noise(mapping)
    solver = _MinimalSolver(target=None, metric=None, compiler=None)
    # each op's section, picked by hand
    sections = {ops.Hadamard: "e", ops.CNOT: "ee", ops.MeasurementCNOTandReset: "ep"}

    checked = 0
    for op in noisy.sequence():
        if isinstance(op, (ops.Input, ops.Output)):
            continue
        expected = solver._identify_noise(type(op), mapping[sections[type(op)]])
        if isinstance(expected, list):
            assert len(op.noise) == len(expected)
            for actual_leg, expected_leg in zip(op.noise, expected):
                assert actual_leg is expected_leg
        else:
            assert op.noise is expected
        checked += 1
    assert checked == 3


def test_assign_noise_with_the_loss_and_depolarizing_benchmark_mapping_reaches_the_cnot_legs():
    circuit = CircuitDAG(n_emitter=2, n_photon=1, n_classical=0)
    circuit.add(ops.CNOT(control=0, control_type="e", target=1, target_type="e"))
    circuit.add(ops.CNOT(control=0, control_type="e", target=0, target_type="p"))

    noisy = circuit.assign_noise(noise_model_loss_and_depolarizing(0.1, 0.05))

    ee_cnot, ep_cnot = [op for op in noisy.sequence() if isinstance(op, ops.CNOT)]
    assert ee_cnot.target_type == "e" and ep_cnot.target_type == "p"
    assert isinstance(ee_cnot.noise[0], nm.DepolarizingNoise)
    assert isinstance(ee_cnot.noise[1], nm.DepolarizingNoise)
    assert isinstance(ep_cnot.noise[0], nm.DepolarizingNoise)
    assert isinstance(ep_cnot.noise[1], nm.NoNoise)


# the five-gate photon wrapper graph_to_circ emits
_FIVE_GATE_WRAPPER = [ops.Hadamard, ops.Phase, ops.Hadamard, ops.Phase, ops.SigmaX]


def _one_photon_wrapper_circuit(operations):
    circuit = CircuitDAG(n_emitter=0, n_photon=1, n_classical=0)
    circuit.add(ops.OneQubitGateWrapper(operations, register=0, reg_type="p"))
    return circuit


def _photon_only_mapping(photon_section):
    return {"e": {}, "p": photon_section, "ee": {}, "ep": {}}


def test_assign_noise_gives_a_wrapper_the_section_wrapper_key_model():
    gate_noise = nm.DepolarizingNoise(0.1)
    wrapper_noise = nm.PhotonLoss(0.1)
    section = {
        "Hadamard": gate_noise,
        "Phase": gate_noise,
        "SigmaX": gate_noise,
        "OneQubitGateWrapper": wrapper_noise,
    }
    circuit = _one_photon_wrapper_circuit(_FIVE_GATE_WRAPPER)

    noisy = circuit.assign_noise(_photon_only_mapping(section))

    assert _the_wrapper(noisy).noise is wrapper_noise


@pytest.mark.parametrize(
    "wrapper_noise",
    [nm.PhotonLoss(0.1), [nm.PhotonLoss(0.1)] + 4 * [nm.NoNoise()]],
    ids=["scalar", "list"],
)
def test_assign_noise_wrapper_key_agrees_with_the_solvers(wrapper_noise):
    section = {
        "Hadamard": nm.DepolarizingNoise(0.1),
        "OneQubitGateWrapper": wrapper_noise,
    }
    solver = _MinimalSolver(target=None, metric=None, compiler=None)
    solver_built = ops.OneQubitGateWrapper(
        _FIVE_GATE_WRAPPER,
        register=0,
        reg_type="p",
        noise=solver._wrap_noise(_FIVE_GATE_WRAPPER, section),
    )

    noisy = _one_photon_wrapper_circuit(_FIVE_GATE_WRAPPER).assign_noise(
        _photon_only_mapping(section)
    )

    assert _the_wrapper(noisy).noise is solver_built.noise
    assert solver_built.noise is wrapper_noise


def test_assign_noise_no_noise_wrapper_key_gives_every_wrapped_gate_no_noise():
    section = {
        "Hadamard": nm.DepolarizingNoise(0.1),
        "OneQubitGateWrapper": nm.NoNoise(),
    }

    noisy = _one_photon_wrapper_circuit(_FIVE_GATE_WRAPPER).assign_noise(
        _photon_only_mapping(section)
    )

    noise = _the_wrapper(noisy).noise
    assert isinstance(noise, list)
    assert len(noise) == len(_FIVE_GATE_WRAPPER)
    assert all(isinstance(model, nm.NoNoise) for model in noise)


def test_assign_noise_wrapper_key_loss_is_applied_once_per_wrapper():
    noisy = _one_photon_wrapper_circuit(_FIVE_GATE_WRAPPER).assign_noise(
        noise_model_pure_loss(0.1)
    )

    compiler = StabilizerCompiler()
    compiler.noise_simulation = True
    state = compiler.compile(circuit=noisy)

    # one loss for the wrapper, not one per lossy gate (0.9 ** 5)
    assert np.isclose(state.rep_data.probability, 0.9)
