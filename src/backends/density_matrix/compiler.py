"""
Compilation tools for simulating a circuit with a purely Density Matrix based backend
"""

import src.backends.density_matrix.functions as dm
from src import ops as ops
from src.backends.compiler_base import CompilerBase
from src.backends.density_matrix.state import DensityMatrix
import src.noise.noise_models as nm


class DensityMatrixCompiler(CompilerBase):
    """
    Compiler which deals exclusively with the DensityMatrix state representation.
    Currently creates a DensityMatrix state object and applies the circuit Operations to it in order

    # TODO: refactor to return a QuantumState object rather than a DensityMatrix object
    # TODO: [longer term] refactor to take a QuantumState object input instead of creating its own initial state?
    """

    name = "density matrix"
    ops = [  # the accepted operations for a given compiler
        ops.Input,
        ops.Identity,
        ops.Phase,
        ops.Hadamard,
        ops.SigmaX,
        ops.SigmaY,
        ops.SigmaZ,
        ops.CNOT,
        ops.CPhase,
        ops.ClassicalCNOT,
        ops.ClassicalCPhase,
        ops.MeasurementZ,
        ops.MeasurementCNOTandReset,
        ops.Output,
    ]

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
        state = state.dm
        if type(op) is ops.Input:
            pass  # TODO: should think about best way to handle inputs/outputs

        elif type(op) is ops.Output:
            pass

        elif type(op) is ops.Identity:
            pass

        elif type(op) is ops.Hadamard:
            unitary = dm.get_one_qubit_gate(
                n_quantum, q_index(op.register, op.reg_type), dm.hadamard()
            )
            state.apply_unitary(unitary)

        elif type(op) is ops.Phase:
            unitary = dm.get_one_qubit_gate(
                n_quantum, q_index(op.register, op.reg_type), dm.phase()
            )
            state.apply_unitary(unitary)

        elif type(op) is ops.SigmaX:
            unitary = dm.get_one_qubit_gate(
                n_quantum, q_index(op.register, op.reg_type), dm.sigmax()
            )
            state.apply_unitary(unitary)

        elif type(op) is ops.SigmaY:
            unitary = dm.get_one_qubit_gate(
                n_quantum, q_index(op.register, op.reg_type), dm.sigmay()
            )
            state.apply_unitary(unitary)

        elif type(op) is ops.SigmaZ:
            unitary = dm.get_one_qubit_gate(
                n_quantum, q_index(op.register, op.reg_type), dm.sigmaz()
            )
            state.apply_unitary(unitary)

        elif type(op) is ops.CNOT:
            unitary = dm.get_two_qubit_controlled_gate(
                n_quantum,
                q_index(op.control, op.control_type),
                q_index(op.target, op.target_type),
                dm.sigmax(),
            )
            state.apply_unitary(unitary)

        elif type(op) is ops.CPhase:
            unitary = dm.get_two_qubit_controlled_gate(
                n_quantum,
                q_index(op.control, op.control_type),
                q_index(op.target, op.target_type),
                dm.sigmaz(),
            )
            state.apply_unitary(unitary)

        elif type(op) is ops.ClassicalCNOT:
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

            classical_registers[op.c_register] = outcome

        elif type(op) is ops.ClassicalCPhase:
            projectors = dm.projectors_zbasis(
                n_quantum, q_index(op.control, op.control_type)
            )

            # apply a Z gate on the target qubit conditioned on the measurement outcome = 1
            unitary = dm.get_one_qubit_gate(
                n_quantum, q_index(op.target, op.target_type), dm.sigmaz()
            )

            outcome = state.apply_measurement_controlled_gate(
                projectors,
                unitary,
                measurement_determinism=self.measurement_determinism,
            )

            classical_registers[op.c_register] = outcome

        elif type(op) is ops.MeasurementCNOTandReset:
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

        elif type(op) is ops.MeasurementZ:
            projectors = dm.projectors_zbasis(
                n_quantum, q_index(op.register, op.reg_type)
            )
            outcome = state.apply_measurement(
                projectors, measurement_determinism=self.measurement_determinism
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
        TODO: consolidate compile_one_gate and compile_one_noisy_gate to one function

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
            op.noise.apply(
                state,
                n_quantum,
                [
                    q_index(op.control, op.control_type),
                    q_index(op.target, op.target_type),
                ],
            )

        # TODO: Handle the following two-qubit noisy gates, currently no replacement or partial replacement
        else:

            if type(op) is ops.ClassicalCNOT:
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
                classical_registers[op.c_register] = outcome

            elif type(op) is ops.ClassicalCPhase:
                projectors = dm.projectors_zbasis(
                    n_quantum, q_index(op.control, op.control_type)
                )
                if isinstance(op.noise, nm.OneQubitGateReplacement):
                    # apply whatever unitary given by the noise model to the target qubit
                    unitary = op.noise.get_backend_dependent_noise(
                        state, n_quantum, [q_index(op.target, op.target_type)]
                    )
                else:
                    # apply a Z gate on the target qubit conditioned on the measurement outcome = 1
                    unitary = dm.get_one_qubit_gate(
                        n_quantum, q_index(op.target, op.target_type), dm.sigmaz()
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
            op.noise.apply(
                state,
                n_quantum,
                [
                    q_index(op.control, op.control_type),
                    q_index(op.target, op.target_type),
                ],
            )
        else:
            raise ValueError(
                f"Noise model not implemented for operation type {type(op)}"
            )
