"""
Compilation tools for simulating a circuit with a Stabilizer backend
"""

from src import ops as ops
from src.backends.compiler_base import CompilerBase
from src.backends.stabilizer.state import MixedStabilizer


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
        state = state.stabilizer

        # TODO: should think about the best way to handle inputs/outputs
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
                print(outcome)
                state.apply_conditioned_gate(
                    q_index(op.target, op.target_type), outcome, gate="x"
                )

            else:
                outcome = state.apply_measurement(
                    q_index(op.control, op.control_type),
                    measurement_determinism=self.measurement_determinism,
                )
                if outcome == 1:
                    state.apply_sigmaz(q_index(op.target, op.target_type))

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

            else:
                outcome = state.apply_measurement(
                    q_index(op.control, op.control_type),
                    measurement_determinism=self.measurement_determinism,
                )

                if outcome == 1:
                    state.apply_sigmax(q_index(op.target, op.target_type))

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
                pass

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
            op.noise.apply(
                state,
                n_quantum,
                [
                    q_index(op.control, op.control_type),
                    q_index(op.target, op.target_type),
                ],
            )
        else:
            pass
