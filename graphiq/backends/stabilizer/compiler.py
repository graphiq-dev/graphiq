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
Compilation tools for simulating a circuit with a Stabilizer backend
"""

from graphiq.backends.compiler_base import CompilerBase
from graphiq.backends.stabilizer.state import MixedStabilizer
from graphiq.circuit import ops as ops


class StabilizerCompiler(CompilerBase):
    """
    Compiler which deals exclusively with the state representation of Stabilizer.
    Currently creates a Stabilizer object and applies the circuit Operations to it in order

    """

    # TODO: [longer term] refactor to take a QuantumState object input instead of creating its own initial state?

    name = "stabilizer"
    ops = [  # the accepted operations for a given compiler
        ops.Input,
        ops.Identity,
        ops.Phase,
        ops.PhaseDagger,
        ops.Hadamard,
        ops.SigmaX,
        ops.SigmaY,
        ops.SigmaZ,
        ops.CNOT,
        ops.CZ,
        ops.ClassicalCNOT,
        ops.ClassicalCZ,
        ops.MeasurementZ,
        ops.MeasurementCNOTandReset,
        ops.Output,
    ]

    def __init__(self, *args, **kwargs):
        """
        Create a compiler which acts on a Stabilizer representation

        :return: nothing
        :rtype: None
        """
        super().__init__(*args, **kwargs)

    def compile_one_gate(self, state, op, n_quantum, q_index, classical_registers):
        """
        Compile one ideal gate

        :param state: the stabilizer representation of the state to be evolved
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
        state = state.rep_data

        if type(op) is ops.Input:
            return

        elif type(op) is ops.Output:
            return

        elif type(op) is ops.Identity:
            return

        elif type(op) is ops.Hadamard:
            state.apply_hadamard(q_index(op.register, op.reg_type))

        elif type(op) is ops.Phase:
            state.apply_phase(q_index(op.register, op.reg_type))

        elif type(op) is ops.PhaseDagger:
            state.apply_phase_dagger(q_index(op.register, op.reg_type))

        elif type(op) is ops.SigmaX:
            state.apply_sigmax(q_index(op.register, op.reg_type))

        elif type(op) is ops.SigmaY:
            state.apply_sigmay(q_index(op.register, op.reg_type))

        elif type(op) is ops.SigmaZ:
            state.apply_sigmaz(q_index(op.register, op.reg_type))

        elif type(op) is ops.CNOT:
            state.apply_cnot(
                control=q_index(op.control, op.control_type),
                target=q_index(op.target, op.target_type),
            )

        elif type(op) is ops.CZ:
            state.apply_cz(
                control=q_index(op.control, op.control_type),
                target=q_index(op.target, op.target_type),
            )

        elif type(op) is ops.ClassicalCNOT:
            # apply an X gate on the target qubit conditioned on the measurement outcome = 1
            if isinstance(state, MixedStabilizer):
                outcome = state.apply_measurement(
                    q_index(op.control, op.control_type),
                    measurement_determinism=self.measurement_determinism,
                )
                state.apply_conditioned_gate(
                    q_index(op.target, op.target_type), outcome, gate="x"
                )
                # outcome has one entry per mixture branch, and branches can
                # disagree; classical_registers only has room for one value
                # per bit, so we report branch 0's outcome. The gate itself
                # is still applied correctly per-branch above. See the TODO
                # on MixedStabilizer.apply_measurement.
                classical_registers[op.c_register] = outcome[0]

            else:
                outcome = state.apply_measurement(
                    q_index(op.control, op.control_type),
                    measurement_determinism=self.measurement_determinism,
                )
                if outcome == 1:
                    state.apply_sigmax(q_index(op.target, op.target_type))
                classical_registers[op.c_register] = outcome

        elif type(op) is ops.ClassicalCZ:
            # apply an Z gate on the target qubit conditioned on the measurement outcome = 1
            if isinstance(state, MixedStabilizer):
                outcome = state.apply_measurement(
                    q_index(op.control, op.control_type),
                    measurement_determinism=self.measurement_determinism,
                )
                state.apply_conditioned_gate(
                    q_index(op.target, op.target_type), outcome, gate="z"
                )
                # see the matching comment in the ClassicalCNOT branch above
                classical_registers[op.c_register] = outcome[0]

            else:
                outcome = state.apply_measurement(
                    q_index(op.control, op.control_type),
                    measurement_determinism=self.measurement_determinism,
                )
                if outcome == 1:
                    state.apply_sigmaz(q_index(op.target, op.target_type))
                classical_registers[op.c_register] = outcome

        elif type(op) is ops.MeasurementCNOTandReset:
            if isinstance(state, MixedStabilizer):
                outcomes = state.apply_measurement(
                    q_index(op.control, op.control_type),
                    measurement_determinism=self.measurement_determinism,
                )
                state.apply_conditioned_gate(
                    q_index(op.target, op.target_type), outcomes, gate="x"
                )
                # see the ClassicalCNOT branch above for why only outcomes[0]
                # is recorded here
                classical_registers[op.c_register] = outcomes[0]

            else:
                outcome = state.apply_measurement(
                    q_index(op.control, op.control_type),
                    measurement_determinism=self.measurement_determinism,
                )

                if outcome == 1:
                    state.apply_sigmax(q_index(op.target, op.target_type))
                classical_registers[op.c_register] = outcome

            # reset the control qubit
            state.reset_qubit(
                q_index(op.control, op.control_type),
                measurement_determinism=self.measurement_determinism,
            )

        elif type(op) is ops.MeasurementZ:
            if isinstance(state, MixedStabilizer):
                outcomes = state.apply_measurement(
                    q_index(op.register, op.reg_type),
                    measurement_determinism=self.measurement_determinism,
                )
                # see the ClassicalCNOT branch above for why only outcomes[0]
                # is recorded here
                classical_registers[op.c_register] = outcomes[0]

            else:
                outcome = state.apply_measurement(
                    q_index(op.register, op.reg_type),
                    measurement_determinism=self.measurement_determinism,
                )
                classical_registers[op.c_register] = outcome

        else:
            raise ValueError(
                f"{type(op)} is invalid or not implemented for {self.__class__.__name__}."
            )

    def compile_one_noisy_gate(
        self, state, op, n_quantum, q_index, classical_registers
    ):
        """
        Compile one noisy gate

        :param state: the QuantumState representation of the state to be evolved,
            where a stabilizer representation can be accessed
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
        # TODO: consolidate compile_one_gate and compile_one_noisy_gate to one function
        if isinstance(op, ops.InputOutputOperationBase):
            pass

        elif isinstance(op, ops.OneQubitOperationBase):
            op.noise.apply(state, n_quantum, [q_index(op.register, op.reg_type)])

        elif isinstance(op, ops.ControlledPairOperationBase):
            op.noise.apply(
                state,
                n_quantum,
                [
                    q_index(op.control, op.control_type),
                    q_index(op.target, op.target_type),
                ],
            )

        # TODO: Handle the following two-qubit noisy gates
        else:
            if type(op) is ops.ClassicalCNOT:
                pass

            elif type(op) is ops.ClassicalCZ:
                pass

            elif type(op) is ops.MeasurementCNOTandReset:
                pass

            elif type(op) is ops.MeasurementZ:
                # refuse rather than silently dropping the measurement, the
                # classical-register write and the noise
                raise NotImplementedError(
                    f"{type(op).__name__} carrying {type(op.noise).__name__} is not "
                    f"implemented for {self.__class__.__name__}."
                )

            else:
                raise ValueError(
                    f"{type(op)} is invalid or not implemented for {self.__class__.__name__}."
                )

    def _apply_additional_noise(self, state, op, n_quantum, q_index):
        """
        A helper function to apply additional noise before or after the operation

        :param state: the state to be applied for the additional noise
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
        elif isinstance(op, ops.ClassicalControlledPairOperationBase):
            # one leg per register, in q_registers order (control first, then
            # target), each on its own qubit -- the same reading as a
            # ControlledPairOperationBase's legs
            for noise, register, reg_type in zip(
                op.noise, op.q_registers, op.q_registers_type
            ):
                noise.apply(state, n_quantum, [q_index(register, reg_type)])
        else:
            raise ValueError(
                f"Noise model not implemented for operation type {type(op)}"
            )
