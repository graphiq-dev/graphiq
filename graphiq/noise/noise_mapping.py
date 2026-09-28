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
A single noise lookup shared by ``SolverBase._identify_noise`` and
``CircuitDAG._noisy_gates``, so the two callers read a noise-model mapping the same way instead
of maintaining two independently drifting implementations.

A noise-model mapping is a nested dict: the outer keys are register-type sections (``"e"``,
``"p"``, ``"ee"``, ``"ep"`` in every production mapping), and each section maps an operation's
class name (optionally suffixed ``_control``/``_target`` for a two-legged op) to a noise model.
"""

import graphiq.circuit.ops as ops
import graphiq.noise.noise_models as nm


def register_section(op) -> str:
    """
    The mapping section an operation instance reads its noise from.

    For a ``ControlledPairOperationBase`` or ``ClassicalControlledPairOperationBase``
    instance this is ``op.control_type + op.target_type`` (e.g. ``"ee"``, ``"ep"``). For every
    other op it is ``op.reg_type``.

    Every production mapping carries exactly the four sections ``"e"``, ``"p"``, ``"ee"``,
    ``"ep"``. A section this function returns that the mapping does not carry (e.g. a
    photon-controlled pair op giving ``"pe"`` or ``"pp"``) is not handled here: the caller's own
    lookup raises ``KeyError``, exactly as ``CircuitDAG._noisy_gates`` did before this lookup was
    shared.

    :param op: an operation instance
    :type op: ops.OperationBase
    :return: the section name
    :rtype: str
    """
    if isinstance(
        op, (ops.ControlledPairOperationBase, ops.ClassicalControlledPairOperationBase)
    ):
        return op.control_type + op.target_type

    return op.reg_type


def noise_for_op(op, section):
    """
    The noise model(s) ``op`` reads out of one mapping section.

    ``op`` may be a class or an instance. For a ``ControlledPairOperationBase`` or
    ``ClassicalControlledPairOperationBase`` (control, target), the per-leg keys
    ``<Op>_control`` / ``<Op>_target`` are read first; the bare ``<Op>`` key is only used as a
    fallback when *neither* per-leg key is present, in which case a scalar value is broadcast to
    both legs and a list value is returned as-is after checking it has length 2. So a section
    that sets a per-leg key for one leg and also sets the bare key does not fall back to the
    bare key for the other leg; that leg is ``NoNoise()``.

    For every other op, only the bare ``<Op>`` key is read; its value (or ``NoNoise()``) is
    returned unchanged.

    This matches what ``SolverBase._identify_noise`` returned before it delegated here for
    bare scalar keys and for per-leg keys, except in three cases for a pair op. A bare list
    value used to come back as a list of two copies of that list; a length-2 list is now
    returned as it is, and any other length raises ``ValueError``. A bare key used to reach both
    legs when every per-leg key present had the value ``NoNoise()``; a present per-leg key now
    wins.

    :param op: an operation, as a class or an instance
    :type op: type[ops.OperationBase] or ops.OperationBase
    :param section: one section of a noise-model mapping (e.g. ``noise_model_mapping["ee"]``)
    :type section: dict
    :return: a noise model, or a list of two for a pair op
    :rtype: nm.NoiseBase or list[nm.NoiseBase]
    :raises ValueError: if a pair op's bare list value does not have length 2
    """
    is_instance = isinstance(op, ops.OperationBase)
    cls = type(op) if is_instance else op
    op_name = cls.__name__

    if issubclass(
        cls, (ops.ControlledPairOperationBase, ops.ClassicalControlledPairOperationBase)
    ):
        op_control = op_name + "_control"
        op_target = op_name + "_target"
        if op_control in section or op_target in section:
            control_noise = section.get(op_control, nm.NoNoise())
            target_noise = section.get(op_target, nm.NoNoise())
            return [control_noise, target_noise]
        if op_name in section:
            value = section[op_name]
            if isinstance(value, list):
                if len(value) != 2:
                    raise ValueError(
                        f"{op_name} noise list must have length 2, got {len(value)}"
                    )
                return value
            return [value, value]
        return [nm.NoNoise(), nm.NoNoise()]

    return section.get(op_name, nm.NoNoise())
