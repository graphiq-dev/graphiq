"""
Compilers takes a circuit description and implements the mapping, given a specific representation of the
underlying quantum state.

The Base class defines an API which all compiler implementations should follow
"""
from abc import ABC, abstractmethod
import logging
import numpy as np

import src.ops as ops
from src.circuit import CircuitBase
import src.noise.noise_models as nm
from src.state import QuantumState


class CompilerBase(ABC):
    """
    Base class for compiler implementations.
    In general, compilers compile circuits using a specific representation(s) for the underlying quantum state
    """

    name = "base"
    ops = [ops.OperationBase]  # the accepted operations for a given compiler

    def __init__(self):
        """
        Initializes CompilerBase fields

        :return: function returns nothing
        :rtype: None
        """
        self._measurement_determinism = "probabilistic"
        self._noise_simulation = False
        self._monte_carlo = False

    @property
    def noise_simulation(self):
        """
        Flag to run circuit simulation using noise or not.

        :return: boolean flag to run noise
        :rtype: boolean
        """
        return self._noise_simulation

    @noise_simulation.setter
    def noise_simulation(self, value):
        """
        Set the flag for running noise.

        :param value: True or False
        :return:
        """
        if isinstance(value, bool):
            self._noise_simulation = value
        else:
            raise ValueError("Noise simulation should be True or False.")

    def compile(self, circuit: CircuitBase, initial_state=None):
        """
        Compiles (i.e. produces an output state) circuit, in the appropriate representation.
        This involves sequentially applying each operation of the circuit on the initial state

        :param circuit: the circuit to compile
        :type circuit: CircuitBase
        :raises ValueError: if there is a circuit Operation which is incompatible with this compiler
        :return: the state produced by the circuit
        :rtype: QuantumState
        """
        # TODO: make this more general, but for now we assume all registers are initialized to |0>

        if initial_state:
            assert isinstance(
                initial_state, QuantumState
            ), "the initial state must be a valid QuantumState object"
            assert (
                initial_state.n_qubits == circuit.n_quantum
            ), "the number of qubits in initial state must match the circuit"
            if self.__class__.name == "density matrix":
                state_data = initial_state.dm.data
            elif self.__class__.name == "stabilizer":
                state_data = initial_state.stabilizer.data
            elif self.__class__.name == "graph":
                state_data = initial_state.graph.data
            else:
                state_data = initial_state.all_representations[0].data
        else:
            state_data = circuit.n_quantum

        state = QuantumState(
            n_qubits=circuit.n_quantum,
            data=state_data,  # initialize to |0...0> state
            representation=self.__class__.name,
            mixed=True if (self._noise_simulation and not self._monte_carlo) else False,
        )

        classical_registers = np.zeros(circuit.n_classical)

        # TODO: support self-defined mapping functions later instead of using the default above
        # Get functions which will map from registers to a unique index
        q_index = CompilerBase.reg_to_index_func(circuit.n_photons)

        # the unwrapping allows us to support Wrapper operation types
        seq = circuit.sequence(unwrapped=True)

        for op in seq:
            if type(op) not in self.ops:
                raise RuntimeError(
                    f"The Operation class {op.__class__.__name__} is not valid with "
                    f"the {self.__class__.__name__} compiler"
                )

            no_noise = not self._noise_simulation
            is_controlled_op = isinstance(
                op, ops.ControlledPairOperationBase
            ) or isinstance(op, ops.ClassicalControlledPairOperationBase)

            if is_controlled_op:
                no_noise = (
                    isinstance(op.noise[0], nm.NoNoise)
                    and isinstance(op.noise[1], nm.NoNoise)
                ) or no_noise
            else:
                no_noise = no_noise or isinstance(op.noise, nm.NoNoise)

            if no_noise:
                self.compile_one_gate(
                    state, op, circuit.n_quantum, q_index, classical_registers
                )

            else:
                if is_controlled_op:
                    if isinstance(op.noise[0], nm.AdditionNoiseBase) and isinstance(
                        op.noise[1], nm.AdditionNoiseBase
                    ):
                        after_control = op.noise[0].noise_parameters["After gate"]
                        after_target = op.noise[1].noise_parameters["After gate"]
                        if after_control and after_target:
                            self.compile_one_gate(
                                state,
                                op,
                                circuit.n_quantum,
                                q_index,
                                classical_registers,
                            )
                            self._apply_additional_noise(
                                state, op, circuit.n_quantum, q_index
                            )
                        elif (not after_control) and (not after_target):
                            self._apply_additional_noise(
                                state, op, circuit.n_quantum, q_index
                            )
                            self.compile_one_gate(
                                state,
                                op,
                                circuit.n_quantum,
                                q_index,
                                classical_registers,
                            )
                        elif after_control and (not after_target):
                            noise_copy = op.noise
                            tmp_noise = [nm.NoNoise, op.noise[1]]
                            op.noise = tmp_noise
                            self._apply_additional_noise(
                                state, op, circuit.n_quantum, q_index
                            )

                            self.compile_one_gate(
                                state,
                                op,
                                circuit.n_quantum,
                                q_index,
                                classical_registers,
                            )
                            tmp_noise = [op.noise[0], nm.NoNoise]

                            op.noise = tmp_noise
                            self._apply_additional_noise(
                                state, op, circuit.n_quantum, q_index
                            )
                            op.noise = noise_copy
                        else:
                            noise_copy = op.noise
                            tmp_noise = [op.noise[0], nm.NoNoise]
                            op.noise = tmp_noise
                            self._apply_additional_noise(
                                state, op, circuit.n_quantum, q_index
                            )
                            self.compile_one_gate(
                                state,
                                op,
                                circuit.n_quantum,
                                q_index,
                                classical_registers,
                            )

                            tmp_noise = [nm.NoNoise, op.noise[1]]
                            op.noise = tmp_noise
                            self._apply_additional_noise(
                                state, op, circuit.n_quantum, q_index
                            )
                            op.noise = noise_copy
                    else:
                        raise ValueError(
                            "We currently do not support different noise positions for one controlled gate."
                        )

                else:
                    if isinstance(op.noise, nm.AdditionNoiseBase):
                        if op.noise.noise_parameters["After gate"]:
                            self.compile_one_gate(
                                state,
                                op,
                                circuit.n_quantum,
                                q_index,
                                classical_registers,
                            )

                            self._apply_additional_noise(
                                state, op, circuit.n_quantum, q_index
                            )
                        else:
                            self._apply_additional_noise(
                                state, op, circuit.n_quantum, q_index
                            )
                            self.compile_one_gate(
                                state,
                                op,
                                circuit.n_quantum,
                                q_index,
                                classical_registers,
                            )
                    elif isinstance(op.noise, nm.ReplacementNoiseBase):
                        self.compile_one_noisy_gate(
                            state, op, circuit.n_quantum, q_index, classical_registers
                        )
                    else:

                        raise ValueError("Noise position is not acceptable.")

        return state

    def validate_ops(self, circuit):
        """
        Verifies that all operations of a circuit are valid for the selected compiler

        :param circuit: the circuit for which we are assessing validity of operations with this compiler
        :type circuit: CircuitBase (or some subclass of it)
        :return: True if all operations are valid, False otherwise
        :rtype: bool
        """
        seq = circuit.sequence()
        valid = True

        for i, op in enumerate(seq):
            if type(op) in self.ops:
                logging.info(
                    f"Operation {i} {type(op).__name__} is valid with {type(self).__name__}"
                )
            else:
                logging.error(
                    f"Error: Operation {i} {type(op).__name__} is not valid with {type(self).__name__}"
                )
                valid = False

        return valid

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

    @staticmethod
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

    @abstractmethod
    def compile_one_gate(self, state, op, n_quantum, q_index, classical_registers):
        raise NotImplementedError("Please select a valid compiler")

    @abstractmethod
    def compile_one_noisy_gate(
        self, state, op, n_quantum, q_index, classical_registers
    ):
        raise NotImplementedError("Please select a valid compiler")

    @abstractmethod
    def _apply_additional_noise(self, op, n_quantum, q_index):
        raise NotImplementedError("Please select a valid compiler")
