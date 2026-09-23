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
import graphiq.benchmarks.circuits as circ
import graphiq.circuit.ops as ops
import graphiq.noise.noise_models as nm
from graphiq.circuit.circuit_dag import CircuitDAG
from graphiq.noise.monte_carlo_noise import McNoiseMap, MonteCarloNoise


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
