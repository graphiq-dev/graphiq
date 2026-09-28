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
import graphiq.noise.noise_models as nm
from graphiq.benchmarks.alternate_circuits import noise_model_loss_and_depolarizing
from graphiq.solvers.solver_base import SolverBase


class _MinimalSolver(SolverBase):
    """A concrete SolverBase subclass with no search behaviour, so that
    ``_identify_noise`` -- which does not read any solver state -- can be called on
    an instance without constructing a full solver."""

    def solve(self, *args):
        raise NotImplementedError


@pytest.fixture
def solver():
    return _MinimalSolver(target=None, metric=None, compiler=None)


def test_per_leg_keys_are_honoured(solver):
    control_noise = nm.DepolarizingNoise(0.1)
    target_noise = nm.DepolarizingNoise(0.2)
    mapping = {"CNOT_control": control_noise, "CNOT_target": target_noise}

    result = solver._identify_noise(ops.CNOT, mapping)

    assert result[0] is control_noise
    assert result[1] is target_noise


def test_one_per_leg_key_leaves_the_other_leg_noiseless(solver):
    control_noise = nm.DepolarizingNoise(0.1)
    mapping = {"MeasurementCNOTandReset_control": control_noise}

    result = solver._identify_noise(ops.MeasurementCNOTandReset, mapping)

    assert result[0] is control_noise
    assert isinstance(result[1], nm.NoNoise)


def test_a_bare_name_key_reaches_both_legs(solver):
    noise = nm.DepolarizingNoise(0.1)
    mapping = {"CNOT": noise}

    result = solver._identify_noise(ops.CNOT, mapping)

    assert len(result) == 2
    assert result[0] is noise
    assert result[1] is noise


def test_an_instance_and_its_class_agree(solver):
    control_noise = nm.DepolarizingNoise(0.1)
    target_noise = nm.DepolarizingNoise(0.2)
    mapping = {"CNOT_control": control_noise, "CNOT_target": target_noise}
    op_instance = ops.CNOT(control=0, target=1)

    result_from_instance = solver._identify_noise(op_instance, mapping)
    result_from_class = solver._identify_noise(ops.CNOT, mapping)

    assert result_from_instance[0] is result_from_class[0] is control_noise
    assert result_from_instance[1] is result_from_class[1] is target_noise


def test_one_qubit_ops_are_unchanged(solver):
    noise = nm.DepolarizingNoise(0.1)

    assert solver._identify_noise(ops.Hadamard, {"Hadamard": noise}) is noise
    assert isinstance(solver._identify_noise(ops.Hadamard, {}), nm.NoNoise)


def test_the_loss_and_depolarizing_benchmark_mapping_reaches_the_cnot_legs(solver):
    noise_model_mapping = noise_model_loss_and_depolarizing(0.1, 0.1)

    both_legs = solver._identify_noise(ops.CNOT, noise_model_mapping["ee"])
    assert isinstance(both_legs[0], nm.DepolarizingNoise)
    assert isinstance(both_legs[1], nm.DepolarizingNoise)

    one_leg = solver._identify_noise(ops.CNOT, noise_model_mapping["ep"])
    assert isinstance(one_leg[0], nm.DepolarizingNoise)
    assert isinstance(one_leg[1], nm.NoNoise)
