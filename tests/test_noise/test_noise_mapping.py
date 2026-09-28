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
from graphiq.noise.noise_mapping import noise_for_op, register_section

# --- register_section ---


def test_register_section_of_a_pair_op_uses_control_and_target_types():
    op = ops.CNOT(control=0, control_type="e", target=1, target_type="e")
    assert register_section(op) == "ee"

    op = ops.CNOT(control=0, control_type="e", target=1, target_type="p")
    assert register_section(op) == "ep"


def test_register_section_of_a_classically_controlled_pair_op():
    op = ops.MeasurementCNOTandReset(
        control=0, control_type="e", target=0, target_type="p", c_register=0
    )
    assert register_section(op) == "ep"


def test_register_section_of_a_one_qubit_op_uses_reg_type():
    op = ops.Hadamard(register=0, reg_type="p")
    assert register_section(op) == "p"


# --- noise_for_op: pair ops (class and instance) ---


def test_noise_for_op_on_a_pair_class_honours_per_leg_keys():
    control_noise = nm.DepolarizingNoise(0.1)
    target_noise = nm.DepolarizingNoise(0.2)
    section = {"CNOT_control": control_noise, "CNOT_target": target_noise}

    result = noise_for_op(ops.CNOT, section)

    assert result[0] is control_noise
    assert result[1] is target_noise


def test_noise_for_op_on_a_pair_instance_honours_per_leg_keys():
    control_noise = nm.DepolarizingNoise(0.1)
    target_noise = nm.DepolarizingNoise(0.2)
    section = {"CNOT_control": control_noise, "CNOT_target": target_noise}
    op = ops.CNOT(control=0, target=1)

    result = noise_for_op(op, section)

    assert result[0] is control_noise
    assert result[1] is target_noise


def test_noise_for_op_on_a_pair_op_bare_key_reaches_both_legs():
    noise = nm.DepolarizingNoise(0.1)
    section = {"CNOT": noise}

    result = noise_for_op(ops.CNOT, section)

    assert len(result) == 2
    assert result[0] is noise
    assert result[1] is noise


def test_noise_for_op_on_a_pair_op_bare_key_with_a_two_element_list_passes_through():
    control_noise = nm.DepolarizingNoise(0.1)
    target_noise = nm.DepolarizingNoise(0.2)
    section = {"CNOT": [control_noise, target_noise]}

    result = noise_for_op(ops.CNOT, section)

    assert result[0] is control_noise
    assert result[1] is target_noise


def test_noise_for_op_on_a_pair_op_bare_key_with_a_three_element_list_raises():
    section = {"CNOT": [nm.NoNoise(), nm.NoNoise(), nm.NoNoise()]}

    with pytest.raises(ValueError):
        noise_for_op(ops.CNOT, section)


def test_noise_for_op_on_a_pair_op_explicit_no_noise_leg_key_takes_precedence_over_the_bare_key():
    bare_noise = nm.DepolarizingNoise(0.1)
    section = {"CNOT": bare_noise, "CNOT_control": nm.NoNoise()}

    result = noise_for_op(ops.CNOT, section)

    assert isinstance(result[0], nm.NoNoise)
    assert isinstance(result[1], nm.NoNoise)


def test_noise_for_op_on_a_one_qubit_op_reads_the_bare_key():
    noise = nm.DepolarizingNoise(0.1)

    assert noise_for_op(ops.Hadamard, {"Hadamard": noise}) is noise
    assert isinstance(noise_for_op(ops.Hadamard, {}), nm.NoNoise)
