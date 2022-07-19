"""
Compilation tools for simulating a circuit with a purely Density Matrix based backend
"""

# from functools import reduce

import numpy as np

import src.backends.density_matrix.functions as dm
from src import ops as ops
from src.backends.compiler_base import CompilerBase
from src.backends.density_matrix.state import DensityMatrix
from src.circuit import CircuitBase
import src.noise.noise_models as nm


# TODO: this is deprecated, as we now only use integers to index the quantum registers (rather than tuples)
# def reg_to_index_func(reg_list):
#     """
#     Returns function which map a (reg, bit) tuple pair into an index between 0 and N - 1
#     This allows the compiler to correctly assign a unique index to each qubit/cbit in each register
#
#     :param reg_list: a list, where reg[i] = <# of qudits/cbits in register i>
#     :type reg_list: list (of ints)
#     :return: A function which maps an input tuple (reg, bit) to an index between
#              0 and N - 1, where N is the total number of elements across all registers
#              (i.e. is the sum of reg)
#     :rtype: function
#     """
#     reg_array = np.array(reg_list)
#     cumulative_num_reg = np.cumsum(reg_array)
#
#     def reg_to_index(reg):
#         # TODO: THIS IS DEPRECATED
#         """ Function which maps (reg, bit) to a unique index """
#         assert isinstance(reg, tuple) and len(reg) == 2, f'Register must be provided as a tuple of length 2'
#         if reg[0] == 0:
#             return reg[1]
#         else:
#             return cumulative_num_reg[reg[0] - 1] + reg[1]
#
#     return reg_to_index


def reg_to_index_func(n_photon):
    """
    Given the number of photonic qubits in the circuit, this returns a function which will match a
    given register number and type (i.e. photonic or emitter) to a matrix index

    :param n_photon: the number of photonic qubits in the system being simulated
    :type n_photon: int
    :return: a function which will map register + register type to a matrix index (zero-indexed)
    :rtype: function
    """

    def reg_to_index(reg, reg_type):
        if reg_type == "p":
            return reg
        elif reg_type == "e":
            return reg + n_photon

    return reg_to_index


class DensityMatrixCompiler(CompilerBase):
    """
    Compiler which deals exclusively with the DensityMatrix state representation.
    Currently creates a DensityMatrix state object and applies the circuit Operations to it in order

    # TODO: refactor to return a QuantumState object rather than a DensityMatrix object
    # TODO: [longer term] refactor to take a QuantumState object input instead of creating its own initial state?
    """

    name = "density_matrix"
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

        :return: function returns nothing
        :rtype: None
        """
        super().__init__(*args, **kwargs)
        self._measurement_determinism = "probabilistic"

    @property
    def measurement_determinism(self):
        """
        Returns the measurement determinism (either it's probabilistic, defaults to 0, or defaults to 1)

        :return: determinism setting
        :rtype: str or int
        """
        return self._measurement_determinism

    @measurement_determinism.setter
    def measurement_determinism(self, measurement_setting):
        """
        Sets the measurement setting with which the compiler simulates a circuit
        (this can be set to "probabilistic", 1, 0)

        :param measurement_setting: if "probabilistic", measurement results are probabilistically selected
                                    if 1, measurement results default to 1 unless the probability of measuring p(1) = 0
                                    if 0, measurement results default to 0 unless the probability of measuring p(0) = 0
        :rtype measurement_setting: str/int
        :return: nothing
        :rtype: None
        """
        if measurement_setting in ["probabilistic", 1, 0]:
            self._measurement_determinism = measurement_setting
        else:
            raise ValueError(
                'Measurement determinism can only be set to "probabilistic", 0, or 1'
            )

    def compile(self, circuit: CircuitBase):
        """
        Compiles (i.e. produces an output state) circuit, in density matrix representation.
        This involves sequentially applying each operation of the circuit on the initial state

        :param circuit: the circuit to compile
        :type circuit: CircuitBase
        :raises ValueError: if there is a circuit Operation which is incompatible with this compiler
        :return: the state produced by the circuit
        :rtype: DensityMatrix

        TODO: return a QuantumState object instead
        """
        # TODO: using just the source nodes doesn't distinguish classical and quantum
        # sources = [x for x in circuit.dag.nodes() if circuit.dag.in_degree(x) == 0]

        # TODO: make this more general, but for now we assume all registers are initialized to |0>
        # initialization of quantum registers
        # init = np.outer(np.array([1, 0]), np.array([1, 0])).astype("complex64")

        # TODO: refactor to be a QuantumState object which contains a density matrix
        # state = DensityMatrix(data=reduce(np.kron, circuit.n_quantum * [init]))

        state = DensityMatrix(data=circuit.n_quantum)
        classical_registers = np.zeros(circuit.n_classical)

        # TODO: support self-defined mapping functions later instead of using the default above
        # Get functions which will map from registers to a unique index
        q_index = reg_to_index_func(circuit.n_photons)

        # the unwrapping allows us to support Wrapper operation types
        seq = circuit.sequence(unwrapped=True)

        for op in seq:
            if type(op) not in self.ops:
                raise RuntimeError(
                    f"The Operation class {op.__class__.__name__} is not valid with "
                    f"the {self.__class__.__name__} compiler"
                )

            if isinstance(op.noise, nm.NoNoise):
                self._compile_one_gate(
                    state, op, circuit.n_quantum, q_index, classical_registers
                )
            else:
                if isinstance(op.noise, nm.AdditionNoiseBase):
                    if op.noise.noise_parameters["After gate"]:
                        self._compile_one_gate(
                            state, op, circuit.n_quantum, q_index, classical_registers
                        )
                        self._apply_additional_noise(
                            state, op, circuit.n_quantum, q_index
                        )
                    else:
                        self._apply_additional_noise(
                            state, op, circuit.n_quantum, q_index
                        )
                        self._compile_one_gate(
                            state, op, circuit.n_quantum, q_index, classical_registers
                        )
                elif isinstance(op.noise, nm.ReplacementNoiseBase):
                    self._compile_one_noisy_gate(
                        state, op, circuit.n_quantum, q_index, classical_registers
                    )
                else:
                    raise ValueError("Noise position is not acceptable.")

        return state

    def _compile_one_gate(self, state, op, n_quantum, q_index, classical_registers):
        """
        Compile one ideal gate

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
        if type(op) is ops.Input:
            pass  # TODO: should think about best way to handle inputs/outputs

        elif type(op) is ops.Output:
            pass

        elif type(op) is ops.Identity:
            pass

        elif type(op) is ops.Hadamard:
            unitary = dm.get_single_qubit_gate(
                n_quantum, q_index(op.register, op.reg_type), dm.hadamard()
            )
            state.apply_unitary(unitary)

        elif type(op) is ops.Phase:
            unitary = dm.get_single_qubit_gate(
                n_quantum, q_index(op.register, op.reg_type), dm.phase()
            )
            state.apply_unitary(unitary)

        elif type(op) is ops.SigmaX:
            unitary = dm.get_single_qubit_gate(
                n_quantum, q_index(op.register, op.reg_type), dm.sigmax()
            )
            state.apply_unitary(unitary)

        elif type(op) is ops.SigmaY:
            unitary = dm.get_single_qubit_gate(
                n_quantum, q_index(op.register, op.reg_type), dm.sigmay()
            )
            state.apply_unitary(unitary)

        elif type(op) is ops.SigmaZ:
            unitary = dm.get_single_qubit_gate(
                n_quantum, q_index(op.register, op.reg_type), dm.sigmaz()
            )
            state.apply_unitary(unitary)

        elif type(op) is ops.CNOT:
            unitary = dm.get_controlled_gate(
                n_quantum,
                q_index(op.control, op.control_type),
                q_index(op.target, op.target_type),
                dm.sigmax(),
            )
            state.apply_unitary(unitary)

        elif type(op) is ops.CPhase:
            unitary = dm.get_controlled_gate(
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
            unitary = dm.get_single_qubit_gate(
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
            unitary = dm.get_single_qubit_gate(
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
            unitary = dm.get_single_qubit_gate(
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

    def _compile_one_noisy_gate(
        self, state, op, n_quantum, q_index, classical_registers
    ):
        """
        Compile one noisy gate
        TODO: consolidate _compile_one_gate and _compile_one_noisy_gate to one function

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

        elif isinstance(op, ops.SingleQubitOperationBase):
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
                    unitary = dm.get_single_qubit_gate(
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
                if isinstance(op.noise, nm.OneQubitGateReplacement):
                    # apply whatever unitary given by the noise model to the target qubit
                    unitary = op.noise.get_backend_dependent_noise(
                        state, n_quantum, [q_index(op.target, op.target_type)]
                    )
                else:
                    # apply a Z gate on the target qubit conditioned on the measurement outcome = 1
                    unitary = dm.get_single_qubit_gate(
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

                if isinstance(op.noise, nm.OneQubitGateReplacement):
                    # apply whatever unitary given by the noise model to the target qubit
                    unitary = op.noise.get_backend_dependent_noise(
                        state, n_quantum, [q_index(op.target, op.target_type)]
                    )
                else:
                    # apply an X gate on the target qubit conditioned on the measurement outcome = 1
                    unitary = dm.get_single_qubit_gate(
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
                # TODO: implement measurement-related error model

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

    def _apply_additional_noise(self, state, op, n_quantum, q_index):
        """
        A helper function to apply additional noise before or after the operation

        :param state: the state representation
        :type state: DensityMatrix
        :param op: the operation associated with the noise
        :type op: OperationBase
        :param n_quantum: the number of qubits
        :type n_quantum: int
        :param q_index: a function that maps register + register type to a matrix index (zero-indexed)
        :type q_index: function
        :return: nothing
        :rtype: None
        """
        if isinstance(op, ops.SingleQubitOperationBase):
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
