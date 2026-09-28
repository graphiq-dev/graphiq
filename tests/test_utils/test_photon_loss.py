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
import pytest

import graphiq.circuit.ops as ops
from graphiq.circuit.circuit_dag import CircuitDAG
from graphiq.noise.noise_models import NoNoise, PhotonLoss
from graphiq.utils.photon_loss import photon_survival_rate


def test_photon_survival_rate_reads_the_control_leg_for_a_photon_control():
    circuit = CircuitDAG(n_emitter=0, n_photon=2)
    circuit.add(
        ops.CNOT(
            control=0,
            control_type="p",
            target=1,
            target_type="p",
            noise=[PhotonLoss(0.5), PhotonLoss(0.1)],
        )
    )

    assert photon_survival_rate(circuit) == pytest.approx([0.5, 0.9])


def test_photon_survival_rate_reads_the_control_leg_of_a_classically_controlled_gate():
    circuit = CircuitDAG(n_emitter=0, n_photon=2, n_classical=1)
    circuit.add(
        ops.MeasurementCNOTandReset(
            control=0,
            control_type="p",
            target=1,
            target_type="p",
            c_register=0,
            noise=[PhotonLoss(0.5), PhotonLoss(0.1)],
        )
    )

    assert photon_survival_rate(circuit) == pytest.approx([0.5, 0.9])


def test_photon_survival_rate_photon_control_with_lossless_target():
    circuit = CircuitDAG(n_emitter=0, n_photon=2)
    circuit.add(
        ops.CZ(
            control=0,
            control_type="p",
            target=1,
            target_type="p",
            noise=[PhotonLoss(0.3), NoNoise()],
        )
    )

    assert photon_survival_rate(circuit) == pytest.approx([0.7, 1])


def test_photon_survival_rate_reads_the_control_leg_when_the_target_is_an_emitter():
    circuit = CircuitDAG(n_emitter=1, n_photon=1)
    circuit.add(
        ops.CNOT(
            control=0,
            control_type="p",
            target=0,
            target_type="e",
            noise=[PhotonLoss(0.2), PhotonLoss(0.4)],
        )
    )

    assert photon_survival_rate(circuit) == pytest.approx([0.8])


def test_photon_survival_rate_reads_the_target_leg_for_a_photon_target():
    circuit = CircuitDAG(n_emitter=1, n_photon=1)
    circuit.add(
        ops.CNOT(
            control=0,
            control_type="e",
            target=0,
            target_type="p",
            noise=[PhotonLoss(0.2), PhotonLoss(0.4)],
        )
    )

    assert photon_survival_rate(circuit) == pytest.approx([0.6])


def test_photon_survival_rate_multiplies_losses_along_the_register():
    circuit = CircuitDAG(n_emitter=1, n_photon=1)
    circuit.add(
        ops.CNOT(
            control=0,
            control_type="e",
            target=0,
            target_type="p",
            noise=PhotonLoss(0.25),
        )
    )
    circuit.add(ops.Hadamard(register=0, reg_type="p", noise=PhotonLoss(0.5)))

    assert photon_survival_rate(circuit) == pytest.approx([0.375])


def test_photon_survival_rate_counts_each_leg_of_an_unwrapped_wrapper():
    circuit = CircuitDAG(n_emitter=0, n_photon=1)
    circuit.add(
        ops.OneQubitGateWrapper(
            [ops.Hadamard, ops.Phase],
            register=0,
            reg_type="p",
            noise=[PhotonLoss(0.5), PhotonLoss(0.5)],
        )
    )

    assert photon_survival_rate(circuit) == pytest.approx([0.25])
