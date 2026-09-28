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
Default gate-duration and noise-parameter data consumed by :mod:`graphiq.noise.time_depend_noise`.

``gate_duration_dict`` gives each operation type it lists a duration of 1 (0 for ``Input`` and ``Output``); types
it does not list, such as ``PhaseDagger`` or ``OneQubitGateWrapper``, have no default duration.
``noise_parameters`` is the default noise-parameter dict, merged into any partial one a caller supplies;
``error_ops`` maps the ``"x"``/``"y"``/``"z"``/``"I"`` noise-type letters to the operation class ``insert_error``
inserts for each. ``equal_time_dict`` builds a gate-duration dict with one duration for every given operation type.
"""

import graphiq.circuit.ops as ops


def equal_time_dict(ops_list, duration):
    """
    Build a gate-duration dict, assigning duration to every operation type in ops_list.

    :param ops_list: operation instances whose types become the dict's keys
    :type ops_list: list[ops.OperationBase]
    :param duration: the time duration assigned to each operation type
    :type duration: int or float
    :return: the gate duration dictionary for the given operations
    :rtype: dict
    """
    duration_dict = {}
    ops_types = [type(op) for op in ops_list]
    assert isinstance(
        duration, (float, int)
    ), "time duration must be a float or integer number"
    for types in ops_types:
        duration_dict[types] = duration
    return duration_dict


gate_duration_dict = {
    ops.CNOT: 1,
    ops.CZ: 1,
    ops.SigmaX: 1,
    ops.SigmaY: 1,
    ops.SigmaZ: 1,
    ops.Hadamard: 1,
    ops.Phase: 1,
    ops.ClassicalCNOT: 1,
    ops.ClassicalCZ: 1,
    ops.MeasurementZ: 1,
    ops.MeasurementCNOTandReset: 1,
    ops.ClassicalControlledPairOperationBase: 1,
    ops.ControlledPairOperationBase: 1,
    ops.Identity: 1,
    ops.Input: 0,
    ops.Output: 0,
}
noise_parameters = {
    "error_rate": 0.05,
    "cut_off_prob": 0.99,
    "criteria": "reg_as_control",
    "noise_type": ["x"],
    "reg_specific_noise": (
        {}
    ),  # {'regtype-reg string':'noise type', etc.} e.g. {'e0': 'x', 'e1': ['y', 'z']}
}  # having a list as noise type would apply both kinds of noises separately to the register

error_ops = {"x": ops.SigmaX, "y": ops.SigmaY, "z": ops.SigmaZ, "I": ops.Identity}
