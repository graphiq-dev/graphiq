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
Compilation tools for simulating a circuit with a purely Density Matrix based backend
"""

import graphiq.backends.density_matrix.functions as dm
import graphiq.noise.noise_models as nm
from graphiq.backends.compiler_base import CompilerBase
from graphiq.backends.density_matrix.state import DensityMatrix
from graphiq.circuit import ops as ops


class DensityMatrixCompiler(CompilerBase):
    """
    Compiler which deals exclusively with the DensityMatrix state representation.
    Currently creates a DensityMatrix state object and applies the circuit Operations to it in order

    """

    # TODO: [longer term] refactor to take a QuantumState object input instead of creating its own initial state?

    name = "dm"
    ops = {  # the accepted operations and the single-qubit action needed for each gate
        ops.Input: lambda: None,
        ops.Identity: lambda: None,
        ops.Phase: lambda: dm.phase(),
        ops.PhaseDagger: lambda: dm.phase_dag(),
        ops.Hadamard: lambda: dm.hadamard(),
        ops.SigmaX: lambda: dm.sigmax(),
        ops.SigmaY: lambda: dm.sigmay(),
        ops.SigmaZ: lambda: dm.sigmaz(),
        ops.ParameterizedOneQubitRotation: (
            lambda theta, phi, lam: dm.parameterized_one_qubit_unitary(theta, phi, lam)
        ),
        ops.ParameterizedControlledRotationQubit: (
            lambda theta, phi, lam: dm.parameterized_one_qubit_unitary(theta, phi, lam)
        ),
        ops.CNOT: lambda: dm.sigmax(),
        ops.CZ: lambda: dm.sigmaz(),
        ops.ClassicalCNOT: lambda: dm.sigmax(),
        ops.ClassicalCZ: lambda: dm.sigmaz(),
        ops.MeasurementZ: lambda: dm.sigmaz(),
        ops.MeasurementCNOTandReset: lambda: dm.sigmax(),
        ops.Output: lambda: None,
    }

    def __init__(self, *args, **kwargs):
        """
        Create a compiler which acts on a DensityMatrix state representation

        :return: nothing
        :rtype: None
        """
        super().__init__(*args, **kwargs)

    def compile_one_gate(self, state, op, n_quantum, q_index, classical_registers):
        """
        Compile one ideal gate

        :param state: the QuantumState representation of the state to be evolved, where a density matrix representation can be accessed
        :type state: QuantumState
        :param op: the operation to be applied
        :type op: OperationBase
        :param n_quantum: the number of qubits
        :type n_quantum: int
        :param q_index: a function that maps register + register type to a matrix index (zero-indexed)
        :type q_index: function
        :param classical_registers: a list of values for classical registers
        :type classical_registers: list
        :return: nothing
        :rtype: None
        """
        assert (
            op.__class__ in self.ops.keys()
        ), f"{op.__class__} is not a valid operation for this compiler"
        state = state.rep_data

        params = op.params

        if isinstance(op, ops.InputOutputOperationBase) or isinstance(op, ops.Identity):
            pass  # TODO: should think about best way to handle inputs/outputs

        elif isinstance(op, ops.OneQubitOperationBase):
            unitary = dm.get_one_qubit_gate(
                n_quantum,
                q_index(op.register, op.reg_type),
                self.ops[op.__class__](*params),
            )
            state.apply_unitary(unitary)

        elif isinstance(op, ops.ControlledPairOperationBase):
            unitary = dm.get_two_qubit_controlled_gate(
                n_quantum,
                q_index(op.control, op.control_type),
                q_index(op.target, op.target_type),
                self.ops[op.__class__](*params),
            )
            state.apply_unitary(unitary)

        elif isinstance(op, ops.ClassicalControlledPairOperationBase):
            projectors = dm.projectors_zbasis(
                n_quantum, q_index(op.control, op.control_type)
            )

            # apply an gate on the target qubit conditioned on the measurement outcome = 1
            unitary = dm.get_one_qubit_gate(
                n_quantum,
                q_index(op.target, op.target_type),
                self.ops[op.__class__](*params),
            )

            outcome = state.apply_measurement_controlled_gate(
                projectors,
                unitary,
                measurement_determinism=self.measurement_determinism,
            )

            classical_registers[op.c_register] = outcome

        elif isinstance(op, ops.MeasurementCNOTandReset):
            projectors = dm.projectors_zbasis(
                n_quantum, q_index(op.control, op.control_type)
            )

            # apply an X gate on the target qubit conditioned on the measurement outcome = 1
            unitary = dm.get_one_qubit_gate(
                n_quantum, q_index(op.target, op.target_type), dm.sigmax()
            )

            outcome = state.apply_measurement_controlled_gate(
                projectors,
                unitary,
                measurement_determinism=self.measurement_determinism,
            )

            # reset the control qubit
            reset_kraus_ops = dm.get_reset_qubit_kraus(
                n_quantum, q_index(op.control, op.control_type)
            )

            classical_registers[op.c_register] = outcome
            state.apply_channel(reset_kraus_ops)

        elif isinstance(op, ops.MeasurementZ):
            projectors = dm.projectors_zbasis(
                n_quantum, q_index(op.register, op.reg_type)
            )
            outcome = state.apply_measurement(
                projectors, measurement_determinism=self.measurement_determinism
            )
            classical_registers[op.c_register] = outcome

        else:
            raise ValueError(
                f"The compile function has an error. "
                f"{type(op)} is a valid operation of the class, but the op was not processed"
            )

    def compile_one_noisy_gate(
        self, state, op, n_quantum, q_index, classical_registers
    ):
        """
        Compile one noisy gate

        :param state: the density matrix representation of the state to be evolved
        :type state: DensityMatrix
        :param op: the operation to be applied
        :type op: OperationBase
        :param n_quantum: the number of qubits
        :type n_quantum: int
        :param q_index: a function that maps register + register type to a matrix index (zero-indexed)
        :type q_index: function
        :param classical_registers: a list of values for classical registers
        :type classical_registers: list
        :return: nothing
        :rtype: None
        """
        if isinstance(op, ops.InputOutputOperationBase):
            pass

        elif isinstance(op, ops.OneQubitOperationBase):
            op.noise.apply(state, n_quantum, [q_index(op.register, op.reg_type)])

        elif isinstance(op, ops.ControlledPairOperationBase):
            op.noise[0].apply(
                state,
                n_quantum,
                [q_index(op.control, op.control_type)],
            )
            op.noise[1].apply(
                state,
                n_quantum,
                [q_index(op.target, op.target_type)],
            )

        # TODO: Handle the following two-qubit noisy gates, currently no replacement or partial replacement
        else:
            if isinstance(op, ops.ClassicalControlledPairOperationBase):
                projectors = dm.projectors_zbasis(
                    n_quantum, q_index(op.control, op.control_type)
                )

                if isinstance(op.noise, nm.OneQubitGateReplacement):
                    # apply whatever unitary given by the noise model to the target qubit
                    unitary = op.noise.get_backend_dependent_noise(
                        state, n_quantum, [q_index(op.target, op.target_type)]
                    )
                else:
                    # apply an X gate on the target qubit conditioned on the measurement outcome = 1
                    unitary = dm.get_one_qubit_gate(
                        n_quantum,
                        q_index(op.target, op.target_type),
                        self.ops[op.__class__],
                    )

                outcome = state.dm.apply_measurement_controlled_gate(
                    projectors,
                    unitary,
                    measurement_determinism=self.measurement_determinism,
                )
                classical_registers[op.c_register] = outcome

            elif type(op) is ops.MeasurementCNOTandReset:
                projectors = dm.projectors_zbasis(
                    n_quantum, q_index(op.control, op.control_type)
                )

                if isinstance(op.noise, nm.OneQubitGateReplacement):
                    # apply whatever unitary given by the noise model to the target qubit
                    unitary = op.noise.get_backend_dependent_noise(
                        state, n_quantum, [q_index(op.target, op.target_type)]
                    )
                else:
                    # apply an X gate on the target qubit conditioned on the measurement outcome = 1
                    unitary = dm.get_one_qubit_gate(
                        n_quantum, q_index(op.target, op.target_type), dm.sigmax()
                    )

                outcome = state.dm.apply_measurement_controlled_gate(
                    projectors,
                    unitary,
                    measurement_determinism=self.measurement_determinism,
                )

                # reset the control qubit
                reset_kraus_ops = dm.get_reset_qubit_kraus(
                    n_quantum, q_index(op.control, op.control_type)
                )

                classical_registers[op.c_register] = outcome
                state.dm.apply_channel(reset_kraus_ops)

            elif type(op) is ops.MeasurementZ:
                # TODO: implement measurement-related error model

                projectors = dm.projectors_zbasis(
                    n_quantum, q_index(op.register, op.reg_type)
                )
                outcome = state.dm.apply_measurement(
                    projectors, measurement_determinism=self.measurement_determinism
                )
                classical_registers[op.c_register] = outcome

            else:
                raise ValueError(
                    f"{type(op)} is invalid or not implemented for {self.__class__.__name__}."
                )

    def _apply_additional_noise(self, state, op, n_quantum, q_index):
        """
        A helper function to apply additional noise before or after the operation

        :param state: the state representation
        :type state: QuantumState
        :param op: the operation associated with the noise
        :type op: OperationBase
        :param n_quantum: the number of qubits
        :type n_quantum: int
        :param q_index: a function that maps register + register type to a matrix index (zero-indexed)
        :type q_index: function
        :return: nothing
        :rtype: None
        """
        if isinstance(op, ops.OneQubitOperationBase):
            op.noise.apply(state, n_quantum, [q_index(op.register, op.reg_type)])
        elif isinstance(op, ops.ControlledPairOperationBase):
            if not isinstance(op.noise, list):
                op.noise = [op.noise] * 2
            control_noise = op.noise[0]
            target_noise = op.noise[1]
            control_noise.apply(
                state, n_quantum, [q_index(op.control, op.control_type)]
            )
            target_noise.apply(state, n_quantum, [q_index(op.target, op.target_type)])
        else:
            raise ValueError(
                f"Noise model not implemented for operation type {type(op)}"
            )
