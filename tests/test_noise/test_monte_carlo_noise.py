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
How MonteCarloNoise draws the noise it attaches to each operation of a circuit: which sub-gate
of a wrapped operation each draw is paired with, a wrapped operation that carries a single model
for the whole wrapper, and the independent draws for the two legs of a controlled gate.
"""

import pytest

import graphiq.circuit.ops as ops
import graphiq.noise.noise_models as nm
from graphiq.circuit.circuit_dag import CircuitDAG
from graphiq.noise.monte_carlo_noise import McNoiseMap, MonteCarloNoise


def test_noisy_gates_maps_a_wrapped_ops_sub_gates_in_constructor_order():
    # OneQubitGateWrapper pairs noise[i] with operations[i], in the order given to the
    # constructor, so _noisy_gates/_find_wrapped_noise draw noise in that order too.
    # Mapping the first two sub-gates at probability 1.0 avoids the residual NoNoise draw that
    # would otherwise make the ordering assertion flaky; the third, unmapped sub-gate is what
    # reaches the NoNoise arm for an op with no entry in the mapping.
    circuit = CircuitDAG(n_emitter=1, n_photon=0, n_classical=0)
    circuit.add(
        ops.OneQubitGateWrapper(
            [ops.Hadamard, ops.SigmaX, ops.SigmaZ], register=0, reg_type="e"
        )
    )

    mc = MonteCarloNoise(circuit)
    mc.n_noisy_gates = 0
    mc_noise_model = McNoiseMap()
    mc_noise_model.add_gate_noise("e", "Hadamard", [(nm.PauliError("X"), 1.0)])
    mc_noise_model.add_gate_noise("e", "SigmaX", [(nm.PauliError("Z"), 1.0)])
    mc.mc_noise_model = mc_noise_model

    noisy_ops = mc._noisy_gates()

    assert len(noisy_ops) == 1
    noise = noisy_ops[0].noise
    # constructor order is operations=[Hadamard, SigmaX, SigmaZ]
    assert (
        isinstance(noise[0], nm.PauliError)
        and noise[0].noise_parameters["Pauli error"] == "X"
    )
    assert (
        isinstance(noise[1], nm.PauliError)
        and noise[1].noise_parameters["Pauli error"] == "Z"
    )
    assert isinstance(noise[2], nm.NoNoise)

    noise_by_gate = {type(gate).__name__: gate.noise for gate in noisy_ops[0].unwrap()}
    assert noise_by_gate["Hadamard"].noise_parameters["Pauli error"] == "X"
    assert noise_by_gate["SigmaX"].noise_parameters["Pauli error"] == "Z"
    assert isinstance(noise_by_gate["SigmaZ"], nm.NoNoise)
    assert mc.n_noisy_gates == 2


def test_noisy_gates_leaves_a_whole_wrapper_model_in_place():
    # a wrapper carrying one model for the whole wrapper keeps it: no per-gate draw replaces
    # it, and none is counted
    circuit = CircuitDAG(n_emitter=1, n_photon=0, n_classical=0)
    circuit.add(
        ops.OneQubitGateWrapper(
            [ops.Hadamard, ops.SigmaX],
            register=0,
            reg_type="e",
            noise=nm.PauliError("X"),
        )
    )

    mc = MonteCarloNoise(circuit)
    mc.n_noisy_gates = 0
    mc_noise_model = McNoiseMap()
    mc_noise_model.add_gate_noise("e", "Hadamard", [(nm.PauliError("Z"), 1.0)])
    mc.mc_noise_model = mc_noise_model

    noisy_ops = mc._noisy_gates()

    assert len(noisy_ops) == 1
    wrapper = noisy_ops[0]
    assert isinstance(wrapper.noise, nm.PauliError)
    assert wrapper.noise.noise_parameters["Pauli error"] == "X"
    assert len(wrapper.unwrap()) == 3
    assert mc.n_noisy_gates == 0


def test_noisy_gates_controlled_gate_legs_can_differ_across_seeds():
    # The controlled-gate noise-LIST arm is unreachable through the public API: the
    # McNoiseMap.mapping setter, which add_gate_noise reaches through add_noise_tuple,
    # requires the first element of every noise tuple to be a NoiseBase, so a list-valued
    # entry is rejected before any draw happens. Only the per-leg
    # redraw in the scalar-noise else arm is coverable -- control and target are drawn
    # independently, so they can land on different noise objects.
    circuit = CircuitDAG(n_emitter=2, n_photon=0, n_classical=0)
    circuit.add(ops.CNOT(control=0, control_type="e", target=1, target_type="e"))
    mc_noise_model = McNoiseMap()
    mc_noise_model.add_gate_noise(
        "ee", "CNOT", [(nm.PauliError("X"), 0.5), (nm.PauliError("Z"), 0.5)]
    )

    def pauli_name(noise):
        return noise.noise_parameters["Pauli error"]

    for seed in range(20):
        mc = MonteCarloNoise(circuit, seed=seed)
        mc.n_noisy_gates = 0
        mc.mc_noise_model = mc_noise_model

        control_noise, target_noise = mc._noisy_gates()[0].noise
        if pauli_name(control_noise) != pauli_name(target_noise):
            return

    pytest.fail("the two legs of a CNOT never differed across 20 seeds")
