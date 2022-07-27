"""
Compilation tools for simulating a circuit with a Stabilizer backend
"""


import numpy as np

from src import ops as ops
from src.backends.compiler_base import CompilerBase
from src.backends.stabilizer.state import Stabilizer

from src.circuit import CircuitBase
import src.noise.noise_models as nm


def reg_to_index_func(n_photon):
    """
    Given the number of photonic qubits in the circuit, this returns a function which will match a
    given register number and type (i.e. photonic or emitter) to a matrix index
    # TODO: this is a common function for several compilers

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


class StabilizerCompiler(CompilerBase):
    """
    Compiler which deals exclusively with the state representation of Stabilizer.
    Currently creates a Stabilizer object and applies the circuit Operations to it in order

    # TODO: refactor to return a QuantumState object rather than a Stabilizer object
    # TODO: [longer term] refactor to take a QuantumState object input instead of creating its own initial state?
    """

    name = "stabilizer"
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
        Create a compiler which acts on a Stabilizer representation

        :return: nothing
        :rtype: None
        """
        super().__init__(*args, **kwargs)
        self._measurement_determinism = "probabilistic"
        self._noise_simulation = True

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
        :type measurement_setting: str or int
        :return: nothing
        :rtype: None
        """
        if measurement_setting in ["probabilistic", 1, 0]:
            self._measurement_determinism = measurement_setting
        else:
            raise ValueError(
                'Measurement determinism can only be set to "probabilistic", 0, or 1'
            )

    @property
    def noise_simulation(self):
        """
        Returns the setting for noise simulation

        :return: the setting for noise simulation
        :rtype: bool
        """
        return self._noise_simulation

    @noise_simulation.setter
    def noise_simulation(self, choice):
        """
        Set the setting for noise simulation

        :param choice: True to enable noise simulation; False to disable noise simulation
        :type choice: bool
        :return: nothing
        :rtype: None
        """
        if type(choice) is bool:
            self._noise_simulation = choice
        else:
            raise ValueError("Noise simulation choice can only be set to True or False")

    def compile(self, circuit: CircuitBase):
        """
        Compiles (i.e. produces an output state) circuit, in stabilizer representation.
        This involves sequentially applying each operation of the circuit on the initial state

        :param circuit: the circuit to compile
        :type circuit: CircuitBase
        :raises ValueError: if there is a circuit Operation which is incompatible with this compiler
        :return: the state produced by the circuit
        :rtype: Stabilizer

        TODO: return a QuantumState object instead
        """
        # TODO: using just the source nodes doesn't distinguish classical and quantum
        # sources = [x for x in circuit.dag.nodes() if circuit.dag.in_degree(x) == 0]

        # TODO: make this more general, but for now we assume all registers are initialized to |0>
        # initialization of quantum registers

        # TODO: refactor to be a QuantumState object which contains stabilizer

        state = Stabilizer(data=circuit.n_quantum)
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

            if not self._noise_simulation or isinstance(op.noise, nm.NoNoise):
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

        :param state: the stabilizer representation of the state to be evolved
        :type state: Stabilizer
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

            state.apply_hadamard(q_index(op.register, op.reg_type))

        elif type(op) is ops.Phase:

            state.apply_phase(q_index(op.register, op.reg_type))

        elif type(op) is ops.SigmaX:
            state.apply_SigmaX(q_index(op.register, op.reg_type))

        elif type(op) is ops.SigmaY:
            state.apply_SigmaY(q_index(op.register, op.reg_type))

        elif type(op) is ops.SigmaZ:
            state.apply_SigmaZ(q_index(op.register, op.reg_type))

        elif type(op) is ops.CNOT:
            state.apply_CNOT(
                control=q_index(op.control, op.control_type),
                target=q_index(op.target, op.target_type),
            )

        elif type(op) is ops.CPhase:
            state.apply_CPhase(
                control=q_index(op.control, op.control_type),
                target=q_index(op.target, op.target_type),
            )

        elif type(op) is ops.ClassicalCNOT:

            # apply an X gate on the target qubit conditioned on the measurement outcome = 1
            outcome = state.apply_measurement(
                q_index(op.control, op.control_type),
                measurement_determinism=self.measurement_determinism,
            )

            if outcome == 1:
                state.apply_SigmaX(q_index(op.target, op.target_type))

            classical_registers[op.c_register] = outcome

        elif type(op) is ops.ClassicalCPhase:
            # apply an Z gate on the target qubit conditioned on the measurement outcome = 1
            outcome = state.apply_measurement(
                q_index(op.control, op.control_type),
                measurement_determinism=self.measurement_determinism,
            )

            if outcome == 1:
                state.apply_SigmaX(q_index(op.target, op.target_type))

            classical_registers[op.c_register] = outcome

        elif type(op) is ops.MeasurementCNOTandReset:
            outcome = state.apply_measurement(
                q_index(op.control, op.control_type),
                measurement_determinism=self.measurement_determinism,
            )

            if outcome == 1:
                state.apply_SigmaX(q_index(op.target, op.target_type))

            # reset the control qubit
            state.reset_qubit(q_index(op.target, op.target_type))

        elif type(op) is ops.MeasurementZ:
            outcome = state.apply_measurement(
                q_index(op.register, op.reg_type),
                measurement_determinism=self.measurement_determinism,
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

        :param state: the stabilizer representation of the state to be evolved
        :type state: Stabilizer
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

        # TODO: Handle the following two-qubit noisy gates
        else:
            if type(op) is ops.ClassicalCNOT:
                pass

            elif type(op) is ops.ClassicalCPhase:
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

        :param state: the Stabilizer representation
        :type state: Stabilizer
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
