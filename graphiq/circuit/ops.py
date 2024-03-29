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
The Operation objects are objects which tell the compiler what gate to apply on which registers / qubits

They serve two purposes:
    1. They are used to build our circuit: circuit.add(OperationObj). The circuit constructs the DAG structure from the
       connectivity information in OperationObj
    2. They are passed as an iterable to the compiler, such that the compiler can perform the correct series of
       information

REGISTERS VS QUDITS/CBITS: following qiskit/openQASM and other software, we allow a distinction between registers
and qudits/qubits (where one register can contain a number of qubits / qudits).

IF an operation is given q_registers=(a, b), the circuit takes it to apply between registers a and b (that is,
it applies between all qubits a[j], b[j] for 0 < j < n with a, b having length n)

If an operation is given q_registers=((a, b), (c, d)), the circuit takes it to apply between qubit b of register a,
and qubit d of register c.

We can also use a mixture of registers and qubits: q_registers=(a, (b, c)) means that the operation will be applied
between EACH QUBIT of register a, and qubit c of register b

"""

import itertools

from abc import ABC

import numpy as np

import graphiq.backends.density_matrix.functions as dmf
import graphiq.noise.noise_models as nm
import graphiq.utils.openqasm_lib as oq_lib

""" Base classes from which operations will inherit """


class OperationBase(ABC):
    """
    Base class from which operations will inherit
    """

    _openqasm_info = None  # This is the information necessary to add our Operation into an openQASM script

    # If _openqasm_info is None, a given operation cannot be added to openQASM

    def __init__(
        self,
        q_registers=tuple(),
        q_registers_type=tuple(),
        c_registers=tuple(),
        noise=nm.NoNoise(),
        params=tuple(),
        param_info=dict(),
    ):
        """
        Creates an Operation base object (which is largely responsible for holding the registers on which
        an operation act--this provides a consistent API for the circuit class to use when dealing with
        any arbitrary operation).

        :param q_registers: (a, ..., b) indices (indicating registers a, ..., b) OR
                            ((a, b), ...) indicating photonic qubit b of register a OR
                            any combination of (reg, bit) notation and reg notation
                            These tuples can be of length 1 or empty as well, depending on the number
                            of registers the gate requires
        :type q_registers: tuple (tuple of: integers OR tuples of length 2)
        :param q_registers_type: tuple of strings, each is either 'p' (photonic qubit) or 'e' (emitter qubit)
        :type q_registers: tuple (of str)
        :param c_registers: same as photon/emitter_registers, but for the classical registers
        :type c_registers: tuple (tuple of: integers OR tuples of length 2)
        :param noise: Noise model
        :type noise: graphiq.noise.noise_models.NoiseBase
        :param params: an ordered list of parameter values for a parameterized gate
        :type params: tuple
        :param param_info: a dictionary that specifies all the information regarding parameters
        :type param_info: dict
        :raises AssertionError: if photon_register, emitter_register, c_registers are not tuples,
                               OR if the elements of the tuple do not correspond to the notation described above
        :return: nothing
        :rtype: None
        """
        assert isinstance(q_registers, tuple)
        assert isinstance(q_registers_type, tuple)
        assert len(q_registers) == len(q_registers_type)
        assert isinstance(c_registers, tuple)

        for q, reg_type in zip(q_registers, q_registers_type):
            assert isinstance(q, int) or (
                isinstance(q, tuple)
                and len(q) == 2
                and isinstance(q[0], int)
                and isinstance(q[1], int)
            ), f"Invalid photon_registers: photon_registers tuple must only contain tuples of length 2 or integers"
            assert reg_type == "e" or reg_type == "p"

        for c in c_registers:
            assert isinstance(c, int) or (
                isinstance(c, tuple)
                and len(c) == 2
                and isinstance(c[0], int)
                and isinstance(c[1], int)
            ), f"Invalid c_register: c_register tuple must only contain tuples of length 2 or integers"

        self._q_registers = q_registers
        self._q_registers_type = q_registers_type
        self._c_registers = c_registers
        self._labels = []
        self.noise = noise
        self.params = params
        self.param_info = param_info

    @classmethod
    def openqasm_info(cls):
        """
        Returns the information needed to generate an openQASM script using this operation
        (needed imports, how to define a gate, how to use a gate), packaged as a single object.
        This is used by the Circuit classes to generate openQASM scripts when needed

        :return: the operation class's openqasm information (possibly None)
        """
        if cls._openqasm_info is None:
            raise ValueError("Operation does not have an openQASM translation")
        return cls._openqasm_info

    @property
    def q_registers(self):
        """
        Returns the quantum registers tuple of the Operation class

        :return: the registers tuple
        :rtype: tuple
        """
        return self._q_registers

    @property
    def q_registers_type(self):
        """
        Returns the quantum registers types ('e' for emitter, 'p' for photons) tuple of the Operation class

        :return: the registers type
        :rtype: tuple
        """
        return self._q_registers_type

    @property
    def c_registers(self):
        """
        Returns the c_registers tuple of the Operation class

        :return: the c_registers_tuple
        :rtype: tuple
        """
        return self._c_registers

    @q_registers.setter
    def q_registers(self, q_reg):
        """
        Allows us to change the emitter_registers object on which an Operation acts
        This should only be used by the circuit class!
        Subclass-specific implementations of this function also update other
        register-related fields (e.g. register, target, control) automatically
        when photon_registers is updated

        :param q_reg: new emitter_register which the operation should target
        :type q_reg: tuple
        :raises ValueError: if the new q_reg object does not match the length of self.emitter_registers (Operations
                           should not have variable register numbers)
        :return: function returns nothing
        :rtype: None
        """
        self._update_q_reg(q_reg)

    @c_registers.setter
    def c_registers(self, c_reg):
        """
        Allows us to change the c_registers object on which an Operation acts
        This should only be used by the circuit class!
        Subclass-specific implementations of this function also update other
        register-related fields (e.g. c_register) automatically
        when c_registers is updated

        :param c_reg: new c_register which the operation should target
        :type c_reg: tuple
        :raises ValueError: if the new q_reg object does not match the length of self.q_registers (Operations
                           should not have variable register numbers)
        :return: function returns nothing
        :rtype: None
        """
        self._update_c_reg(c_reg)

    @q_registers_type.setter
    def q_registers_type(self, q_regs_type):
        self._q_registers_type = q_regs_type

    @property
    def labels(self):
        """
        Returns the list of labels of this operation

        :return: list of labels
        :rtype: list[str]
        """
        return self._labels

    @labels.setter
    def labels(self, new_labels):
        """
        Sets the labels to be new_labels

        :return: nothing
        :rtype: None
        """
        self._labels = new_labels

    def add_labels(self, new_labels):
        """
        Adds new labels to existing labels

        :return: nothing
        :rtype: None
        """
        if isinstance(new_labels, list):
            self._labels += new_labels
        else:
            self._labels.append(new_labels)

    def unwrap(self):
        """
        Unwraps the Operation into a list of sub-operations (this can be useful for any operations which are composed
        of multiple other operations) in the reverse order, which corresponds to the order of applying operations.

        :return: a sequence of base operations (i.e. operations which are not compositions of other operations)
        :rtype: list
        """
        return [self][::-1]

    def _update_q_reg(self, q_reg):
        """
        Helper function to verify the validity of the new q_registers tuple and to update the fields
        This is broken into a separate function because subclasses will need to use this as well,
        and using the super() keyword gets messy with class properties

        :param q_reg: new emitter_register which the operation should target
        :type q_reg: tuple
        :raises ValueError: if the new q_reg object does not match the length of self.emitter_registers (Operations
                           should not have variable register numbers)
        :return: function returns nothing
        :rtype: None
        """
        if len(q_reg) != len(self._q_registers):
            raise ValueError(
                f"The number of quantum registers on which the operation acts cannot be changed!"
            )
        self._q_registers = q_reg

    def _update_c_reg(self, c_reg):
        """
        Helper function to verify the validity of the new c_registers tuple and to update the fields
        This is broken into a separate function because subclasses will need to use this as well,
        and using the super() keyword gets messy with class properties

        :param c_reg: new c_register which the operation should target
        :type c_reg: tuple
        :raises ValueError: if the new c_reg object does not match the length of self.c_registers (Operations
                           should not have variable register numbers)
        :return: function returns nothing
        :rtype: None
        """
        if len(c_reg) != len(self._c_registers):
            raise ValueError(
                f"The number of classical registers on which the operation acts cannot be changed!"
            )
        self._c_registers = c_reg

    def parse_q_reg_types(self):
        """
        Find a proper string description of the register types relevant for this operation

        :raises ValueError: if the quantum register type is not supported
        :return: a string description
        :rtype: str
        """
        type_description = ""
        for i in range(len(self.q_registers_type)):
            if self.q_registers_type[i] == "e":
                type_description += "Emitter-"
            elif self.q_registers_type[i] == "p":
                type_description += "Photonic-"
            else:
                raise ValueError("Detected a non-supported quantum register type.")

        return type_description[:-1]

    @staticmethod
    def _validate_param_info(param_info, n_params):
        """
        Validate the param_info

        :param param_info: param_info for a gate
        :type param_info: None or dict
        :param n_params: number of parameters
        :type n_params: int
        :return: whether the param_info is valid
        :rtype: bool
        """
        if param_info is None:
            return True
        else:
            if not isinstance(param_info, dict):
                return False
            else:
                if ("bounds" not in param_info.keys()) or (
                    "labels" not in param_info.keys()
                ):
                    return False
                else:
                    return (
                        len(param_info["bounds"]) == n_params
                        and len(param_info["labels"]) == n_params
                    )


class OneQubitOperationBase(OperationBase):
    """
    This is used as a base class for any one-qubit operation (one-qubit operations should
    all depend on a single parameter, "register"
    """

    def __init__(self, register, reg_type, noise=nm.NoNoise()):
        """
        Creates a one-qubit operation base class object

        :param register: the (quantum) register on which the single-qubit operation acts
        :type register: int OR tuple (of ints, length 2)
        :param reg_type: 'e' if emitter qubit, 'p' if a photonic qubit
        :type reg_type: str
        :param noise: Noise model
        :type noise: graphiq.noise.noise_models.NoiseBase
        :return: nothing
        :rtype: None
        """
        super().__init__(
            q_registers=(register,), q_registers_type=(reg_type,), noise=noise
        )
        self.register = register
        self.reg_type = reg_type
        self.add_labels("one-qubit")

    @OperationBase.q_registers.setter
    def q_registers(self, q_reg):
        """
        Handle to modify the register-qubit pairs on which the operation acts. This also automatically updates the
        self.register field

        :param q_reg: the new q_register value to set
        :raises ValueError: if the new q_reg object does not have a length of 1
        :return: nothing
        :rtype: None
        """
        self._update_q_reg(q_reg)
        self.register = q_reg[0]


class InputOutputOperationBase(OperationBase):
    """
    This is used as a base class for our Input and Output Operations. These operations largely act as "dummy Operations"
    signalling the input / output of a state (useful for circuit DAG representation, for example)
    """

    # IO Operations don't need openqasm representations, since they are dummy gates useful to circuit representation
    # and do not actually modify the circuit output
    _openqasm_info = oq_lib.empty_info()

    def __init__(self, register, reg_type, noise=nm.NoNoise()):
        """
        Creates an IO base class Operation

        :param register: the register/qubit which this I/O operation acts on
        :type register: int OR tuple (of ints, length 2)
        :param reg_type: the input/output is for a quantum photonic register if 'p',
                         for a quantum emitter register if 'e',
                         and for a classical register if 'c'
        :type reg_type: str
        :param noise: Noise model
        :type noise: graphiq.noise.noise_models.NoiseBase
        :return: nothing
        :rtype: None
        """
        if reg_type == "p" or reg_type == "e":
            super().__init__(
                q_registers=(register,), q_registers_type=(reg_type,), noise=noise
            )
        elif reg_type == "c":
            super().__init__(c_registers=(register,), noise=noise)
        else:
            raise ValueError(
                "Register type must be either quantum photonic (reg_type='p'), "
                "quantum emitter (reg_type='e'), or classical (reg_type='c')"
            )
        self.reg_type = reg_type
        self.register = register

    @OperationBase.q_registers.setter
    def q_registers(self, q_reg):
        """
        Handle to modify the register-qubit pairs on which the operation acts. This also automatically updates the
        self.register field, if the I/O is quantum

        :param q_reg: the new q_register value to set (if any)
        :raises ValueError: if the new q_reg object does not match the length of self.q_registers (Operations
                           should not have variable register numbers)
        :return: function returns nothing
        :rtype: None
        """
        self._update_q_reg(q_reg)
        self.register = q_reg[0]

    @OperationBase.c_registers.setter
    def c_registers(self, c_reg):
        """
        Handle to modify the register-cbit pairs on which the operation acts. This also automatically updates the
        self.register field, if the I/O is classical

        :param c_reg: the new c_register value to set (if any)
        :raises ValueError: if the new c_reg object does not match the length of self.c_registers (Operations
                           should not have variable register numbers)
        :return: function returns nothing
        :rtype: None
        """
        self._update_c_reg(c_reg)
        if self.reg_type == "c":
            self.register = c_reg[0]


class ControlledPairOperationBase(OperationBase):
    """
    This is used as a base class for our quantum controlled gates (e.g. CNOT, CPHASE). Each ControlledPairOperationBase
    should have control and target registers/qubits specified
    """

    def __init__(self, control, control_type, target, target_type, noise=nm.NoNoise()):
        """
        Creates a control gate object

        :param control: control register/qubit for the Operation
        :type control: int OR tuple (of ints, length 2)
        :param control_type: 'p' if photonic qubit, 'e' if emitter qubit
        :type control_type: str
        :param target: target register/qubit for the Operation
        :type target: int OR tuple (of ints, length 2)
        :param target_type: 'p' if photonic qubit, 'e' if emitter qubit
        :type target_type: str
        :return: function returns nothing
        :rtype: None
        """

        if isinstance(noise, list):
            assert len(noise) == 2
        else:
            noise = 2 * [noise]
        super().__init__(
            q_registers=(control, target),
            q_registers_type=(control_type, target_type),
            noise=noise,
        )

        self.control = control
        self.control_type = control_type
        self.target = target
        self.target_type = target_type
        self.add_labels("two-qubit")

    @OperationBase.q_registers.setter
    def q_registers(self, q_reg):
        """
        Handle to modify the register-qubit pairs on which the operation acts. This also automatically updates the
        self.control, self.target fields

        :param q_reg: the new q_register value to set
        :raises ValueError: if the new q_reg object does not have a length of 2
        :return: function returns nothing
        :rtype: None
        """
        self._update_q_reg(q_reg)

        self.control = q_reg[0]
        self.target = q_reg[1]


class ClassicalControlledPairOperationBase(OperationBase):
    """
    This is used as a base class for our classical controlled gates (e.g. classical CNOT, classical CPHASE).
    Each ClassicalControlledPairOperationBase should have control and target registers/qubits specified, and
    a c_register target register/cbit specified.
    """

    def __init__(
        self,
        control,
        control_type,
        target,
        target_type,
        c_register=0,
        noise=nm.NoNoise(),
    ):
        """
        Creates the classically controlled gate

        :param control: the control register/qubit
        :type control: int OR tuple (of ints, length 2)
        :param control_type: 'p' if photonic qubit, 'e' if emitter qubit
        :type control_type: str
        :param target: target register/qubit for the Operation
        :type target: int OR tuple (of ints, length 2)
        :param target_type: 'p' if photonic qubit, 'e' if emitter qubit
        :param c_register: the classical register/cbit
        :type c_register: int OR tuple (of ints, length 2)
        :param noise: Noise model
        :type noise: graphiq.noise.noise_models.NoiseBase
        :return: nothing
        :rtype: None
        """
        if isinstance(noise, list):
            assert len(noise) == 2
        else:
            noise = 2 * [noise]
        super().__init__(
            q_registers=(control, target),
            q_registers_type=(control_type, target_type),
            c_registers=(c_register,),
            noise=noise,
        )

        self.control = control
        self.control_type = control_type
        self.target = target
        self.target_type = target_type
        self.c_register = c_register
        self.add_labels("two-qubit")

    @OperationBase.q_registers.setter
    def q_registers(self, q_reg):
        """
        Handle to modify the register-qubit pairs on which the operation acts. This also automatically updates the
        self.control, self.target fields

        :param q_reg: the new q_register value to set
        :raises ValueError: if the new q_reg object does not have a length of 2
        :return: function returns nothing
        :rtype: None
        """
        self._update_q_reg(q_reg)
        self.control = q_reg[0]
        self.target = q_reg[1]

    @OperationBase.c_registers.setter
    def c_registers(self, c_reg):
        """
        Handle to modify the register-cbit pair on which the operation acts. This also automatically updates the
        self.c_register field

        :param c_reg: the new c_register value to set
        :raises ValueError: if the new c_reg object does not have a length of 1
        :return: function returns nothing
        :rtype: None
        """
        self._update_c_reg(c_reg)
        self.c_register = c_reg[0]


""" Quantum gates """


class OneQubitGateWrapper(OneQubitOperationBase):
    """
    This wrapper class allows us to compose a list of one-qubit operation and treat them as a single component
    within the circuit (this allows us, for example, to create every local Clifford gate with other gates, without
    having to separately implement every combination within the compiler)
    """

    def __init__(self, operations: list, register=0, reg_type="e", noise=nm.NoNoise()):
        if isinstance(noise, nm.NoNoise):
            noise = len(operations) * [nm.NoNoise()]
        super().__init__(register, reg_type, noise)
        if len(operations) == 0:
            raise ValueError(
                "Operation list for the single qubit gate wrapper must be of length 1 or more"
            )
        for op_class in operations:
            assert issubclass(op_class, OneQubitOperationBase)
            # can only contain base classes
            assert not isinstance(op_class, OneQubitGateWrapper)
        self.operations = operations
        self._openqasm_info = oq_lib.single_qubit_wrapper_info(operations)

    def unwrap(self):
        """
        Unwraps the Operation into a list of sub-operations (this can be useful for any operations which are composed
        of multiple other operations) in the reverse order, which corresponds to the order of applying gates

        :return: a sequence of base operations (i.e. operations which are not compositions of other operations)
        :rtype: list
        """
        if isinstance(self.noise, list):
            assert len(self.noise) == len(self.operations)

            gates = [
                self.operations[i](
                    register=self.register, reg_type=self.reg_type, noise=self.noise[i]
                )
                for i in range(len(self.operations))
            ]
        else:
            gates = [
                self.operations[i](
                    register=self.register, reg_type=self.reg_type, noise=nm.NoNoise()
                )
                for i in range(len(self.operations))
            ]
            noise = Identity(
                register=self.register, reg_type=self.reg_type, noise=self.noise
            )
            if self.noise.noise_parameters["After gate"]:
                gates.insert(0, noise)
            else:
                gates.append(noise)
        return gates[::-1]

    def openqasm_info(self):
        return self._openqasm_info


class Hadamard(OneQubitOperationBase):
    """
    Hadamard gate Operation
    """

    _openqasm_info = oq_lib.hadamard_info()

    def __init__(self, register=0, reg_type="e", noise=nm.NoNoise()):
        super().__init__(register, reg_type, noise)


class SigmaX(OneQubitOperationBase):
    """
    Pauli X gate Operation
    """

    _openqasm_info = oq_lib.sigma_x_info()

    def __init__(self, register=0, reg_type="e", noise=nm.NoNoise()):
        super().__init__(register, reg_type, noise)


class SigmaY(OneQubitOperationBase):
    """
    Pauli Y gate Operation
    """

    _openqasm_info = oq_lib.sigma_y_info()

    def __init__(self, register=0, reg_type="e", noise=nm.NoNoise()):
        super().__init__(register, reg_type, noise)


class SigmaZ(OneQubitOperationBase):
    """
    Pauli Z gate Operation
    """

    _openqasm_info = oq_lib.sigma_z_info()

    def __init__(self, register=0, reg_type="e", noise=nm.NoNoise()):
        super().__init__(register, reg_type, noise)


class Phase(OneQubitOperationBase):
    """
    Phase gate operation, P = diag(1, i)
    """

    _openqasm_info = oq_lib.phase_info()

    def __init__(self, register=0, reg_type="e", noise=nm.NoNoise()):
        super().__init__(register, reg_type, noise)


class PhaseDagger(OneQubitOperationBase):
    """
    Phase gate operation, P_dag = diag(1, -i)
    """

    _openqasm_info = oq_lib.phase_dagger_info()

    def __init__(self, register=0, reg_type="e", noise=nm.NoNoise()):
        super().__init__(register, reg_type, noise)


class ParameterizedOneQubitRotation(OneQubitOperationBase):
    """
    Parameterized one qubit rotation.
    """

    _openqasm_info = (
        oq_lib.parameterized_info()
    )  # todo, change to appropriate openQASM info

    def __init__(
        self, register=0, reg_type="e", noise=nm.NoNoise(), params=None, param_info=None
    ):
        super().__init__(register, reg_type, noise)

        if params is None:
            params = (0.0, 0.0, 0.0)

        else:
            if len(params) != 3:
                raise ValueError("Length of params must be 3")

        if param_info is None:
            param_info = {
                "bounds": ((-np.pi, np.pi), (-np.pi, np.pi), (-np.pi, np.pi)),
                "labels": ("theta", "phi", "lambda"),
            }
        else:
            if not self._validate_param_info(param_info, 3):
                raise ValueError("The data format of param_info is invalid.")

        self.params = params
        self.param_info = param_info


class ParameterizedControlledRotationQubit(ControlledPairOperationBase):
    """
    Parameterized two qubit controlled gate,
    """

    _openqasm_info = (
        oq_lib.cparameterized_info()
    )  # todo, change to appropriate openQASM info

    def __init__(
        self,
        control=0,
        control_type="e",
        target=0,
        target_type="e",
        noise=nm.NoNoise(),
        params=None,
        param_info=None,
    ):
        super().__init__(control, control_type, target, target_type, noise)
        if params is None:
            params = (0.0, 0.0, 0.0)
        else:
            if len(params) != 3:
                raise ValueError("Length of params must be 3")

        if param_info is None:
            param_info = {
                "bounds": ((-np.pi, np.pi), (-np.pi, np.pi), (-np.pi, np.pi)),
                "labels": ("theta", "phi", "lambda"),
            }
        else:
            if not self._validate_param_info(param_info, 3):
                raise ValueError("The data format of param_info is invalid.")

        self.params = params
        self.param_info = param_info


class RY(ParameterizedOneQubitRotation):
    """
    Rotation around the Y axis
    """

    _openqasm_info = oq_lib.ry_info()

    def __init__(
        self, register=0, reg_type="e", noise=nm.NoNoise(), params=None, param_info=None
    ):
        super().__init__(register, reg_type, noise)

        if params is None:
            params = (0.0,)
        else:
            if len(params) != 1:
                raise ValueError("Length of params must be 1")
        if param_info is None:
            param_info = {
                "bounds": (-np.pi, np.pi),
                "labels": "theta",
            }
        else:
            if not self._validate_param_info(param_info, 1):
                raise ValueError("The data format of param_info is invalid.")

        self.params = params
        self.param_info = param_info


class RX(ParameterizedOneQubitRotation):
    """
    Rotation around the X axis
    """

    _openqasm_info = oq_lib.rx_info()

    def __init__(
        self, register=0, reg_type="e", noise=nm.NoNoise(), params=None, param_info=None
    ):
        super().__init__(register, reg_type, noise)

        if params is None:
            params = (0.0,)

        else:
            if len(params) != 1:
                raise ValueError("Length of params must be 1")

        if param_info is None:
            param_info = {
                "bounds": (-np.pi, np.pi),
                "labels": "theta",
            }
        else:
            if not self._validate_param_info(param_info, 1):
                raise ValueError("The data format of param_info is invalid.")

        self.params = params
        self.param_info = param_info


class RZ(ParameterizedOneQubitRotation):
    """
    Rotation around the Z axis
    """

    _openqasm_info = oq_lib.rz_info()

    def __init__(
        self, register=0, reg_type="e", noise=nm.NoNoise(), params=None, param_info=None
    ):
        super().__init__(register, reg_type, noise)

        if params is None:
            params = (0.0,)

        else:
            if len(params) != 1:
                raise ValueError("Length of params must be 1")
        if param_info is None:
            param_info = {
                "bounds": (-np.pi, np.pi),
                "labels": "phi",
            }
        else:
            if not self._validate_param_info(param_info, 1):
                raise ValueError("The data format of param_info is invalid.")

        self.params = params
        self.param_info = param_info


class Identity(OneQubitOperationBase):
    """
    Identity Operation
    """

    _openqasm_info = oq_lib.empty_info()

    def __init__(self, register=0, reg_type="e", noise=nm.NoNoise()):
        super().__init__(register, reg_type, noise)


class CNOT(ControlledPairOperationBase):
    """
    CNOT gate Operation
    """

    _openqasm_info = oq_lib.cnot_info()

    def __init__(
        self, control=0, control_type="e", target=0, target_type="e", noise=nm.NoNoise()
    ):
        super().__init__(control, control_type, target, target_type, noise)


class CZ(ControlledPairOperationBase):
    """
    Controlled-Z gate Operation
    """

    _openqasm_info = oq_lib.cz_info()

    def __init__(
        self, control=0, control_type="e", target=0, target_type="e", noise=nm.NoNoise()
    ):
        super().__init__(control, control_type, target, target_type, noise)


class ClassicalCNOT(ClassicalControlledPairOperationBase):
    """
    Classical CNOT gate Operation
    """

    _openqasm_info = oq_lib.classical_cnot_info()


class ClassicalCZ(ClassicalControlledPairOperationBase):
    """
    Classical CZ gate Operation
    """

    _openqasm_info = oq_lib.classical_cz_info()


class MeasurementCNOTandReset(ClassicalControlledPairOperationBase):
    """
    Measurement-controlled X gate Operation with resetting the control qubit after measurement
    """

    _openqasm_info = oq_lib.measurement_cnot_and_reset()

    def __init__(
        self,
        control: object = 0,
        control_type: object = "e",
        target: object = 0,
        target_type: object = "p",
        c_register: object = 0,
        noise: object = nm.NoNoise(),
    ) -> object:
        super().__init__(control, control_type, target, target_type, c_register, noise)


class MeasurementZ(OperationBase):
    """
    Z Measurement Operation
    """

    # TODO: maybe create a base class for measurements in the future IFF we also want to support other measurements

    _openqasm_info = oq_lib.z_measurement_info()

    def __init__(self, register=0, reg_type="e", c_register=0, noise=nm.NoNoise()):
        """
        Creates a Z measurement Operation

        :param register: the quantum register on which the measurement is performed
        :type register: int OR tuple (of ints, length 2)
        :param reg_type: 'p' if photonic qubit, 'e' if emitter qubit
        :type reg_type: str
        :param c_register: the classical register to which the measurement result is saved
        :type c_register: int OR tuple (of ints, length 2)
        :param noise: Noise model
        :type noise: src.noise.noise_models.NoiseBase
        :return: this function returns nothing
        :rtype: None
        """
        super().__init__(
            q_registers=(register,),
            q_registers_type=(reg_type,),
            c_registers=(c_register,),
            noise=noise,
        )

        self.register = register
        self.reg_type = reg_type
        self.c_register = c_register
        self.add_labels("one-qubit")

    @OperationBase.q_registers.setter
    def q_registers(self, q_reg):
        """
        Handle to modify the register-qubit pair which is measured. This also automatically updates the
        self.register field

        :param q_reg: the new q_register value to set
        :raises ValueError: if the new q_reg object does not have a length of 1
        :return: function returns nothing
        :rtype: None
        """
        self._update_q_reg(q_reg)
        self.register = q_reg[0]

    @OperationBase.c_registers.setter
    def c_registers(self, c_reg):
        """
        Handle to modify the register-cbit pair to which measurements are saved. This also automatically updates the
        self.c_register field

        :param c_reg: the new c_register value to set
        :raises ValueError: if the new c_reg object does not have a length of 1
        :return: function returns nothing
        :rtype: None
        """
        self._update_c_reg(c_reg)
        self.c_register = c_reg[0]


class Input(InputOutputOperationBase):
    """
    Input Operation. Serves as a placeholder in the circuit so that we know that this is where a given
    qubit/cbit is introduced (i.e. there are no prior operations on it)
    """

    def __init__(self, register=None, reg_type="e"):
        super().__init__(register, reg_type=reg_type)


class Output(InputOutputOperationBase):
    """
    Input Operation. Serves as a placeholder in the circuit so that we know that this is the final operation on a
    qubit/cbit (i.e. there are no subsequent operations on it)
    """

    def __init__(self, register=None, reg_type="e"):
        super().__init__(register, reg_type=reg_type)


""" Helper function to get useful sets of gates """


def local_clifford_composition():
    a = [
        [Identity],
        [Hadamard, Phase, Hadamard, Phase],
        [Hadamard, Phase],
        [Hadamard],
        [Phase, Hadamard, Phase],
        [Phase],
    ]
    b = [[Identity], [SigmaX], [SigmaY], [SigmaZ]]
    return a, b


def one_qubit_cliffords():
    """
    Returns an iterator of single-qubit clifford gates

    :return: iterator covering each single-qubit clifford gate
    :rtype: map
    """
    a, b = local_clifford_composition()

    def flatten_gates(c):
        return c[0] + c[1]  # where c is a tuple of lists

    return map(flatten_gates, itertools.product(a, b))


def local_clifford_to_matrix_map(gate):
    """
    Find the 2 X 2 matrix corresponding to the local Clifford gate

    :param gate: the local Clifford gate
    :type gate: list or a subclass of OperationBase
    :return: the 2 X 2 matrix corresponding to the local Clifford gate
    :rtype: numpy.ndarray
    """
    mapping = {
        Identity.__name__: np.eye(2),
        Hadamard.__name__: dmf.hadamard(),
        Phase.__name__: dmf.phase(),
        SigmaX.__name__: dmf.sigmax(),
        SigmaY.__name__: dmf.sigmay(),
        SigmaZ.__name__: dmf.sigmaz(),
    }

    if isinstance(gate, list):
        result = np.eye(2)
        for op in gate:
            if op.__name__ in mapping.keys():
                result = result @ mapping[op.__name__]
            else:
                raise ValueError(f"Cannot support the operator of type {op.__name__}")
        return result
    else:
        if gate.__name__ in mapping.keys():
            return mapping[gate.__name__]
        else:
            raise ValueError(f"Cannot support the operator of type {gate.__name__}")


def local_cliffords_name_to_matrix_map():
    """
    Find all one-qubit local Clifford gates in the matrix representation

    :return: all one-qubit local Clifford gates in the matrix representation
    :rtype: map
    """
    a, b = local_clifford_composition()

    def gate_matrix(c):
        # where c is a tuple of lists
        matrix1 = local_clifford_to_matrix_map(c[0])
        matrix2 = local_clifford_to_matrix_map(c[1])

        return matrix1 @ matrix2

    return map(gate_matrix, itertools.product(a, b))


def find_local_clifford_by_matrix(matrix):
    """
    Find local Clifford by its matrix representation

    :param matrix: a matrix representation of one-qubit Clifford gate
    :type matrix: numpy.ndarray
    :raises ValueError: if the matrix does not correspond to a valid one-qubit Clifford gate.
    :return: the local Clifford gate specified by a list of basic gates it consists of
        or None if the input matrix is not a valid one-qubit Clifford gate
    :rtype: list
    """
    gate_list1, gate_list2 = local_clifford_composition()
    for op1 in gate_list1:
        for op2 in gate_list2:
            matrix1 = local_clifford_to_matrix_map(op1)
            matrix2 = local_clifford_to_matrix_map(op2)
            product_matrix = matrix1 @ matrix2
            if dmf.check_equivalent_unitaries(matrix, product_matrix):
                return op1 + op2

    raise ValueError("Invalid one-qubit Clifford gate.")


def simplify_local_clifford(gate_list):
    """
    Simplify the list of basic gates that represents a local Clifford

    :param gate_list: original list of basic gates that represents a local Clifford
    :type gate_list: list
    :return: simplified list of basic gates that represents the same local Clifford
    :rtype: list
    """
    matrix = local_clifford_to_matrix_map(gate_list)

    return find_local_clifford_by_matrix(matrix)


def name_to_class_map(name):
    """
    Maps our openqasm naming scheme to operation classes. Does not handle multi-gate wrappers
    Does not handle multi-line openqasm components

    :param name: gate name in openqasm
    :type name: str
    :return: the operation class corresponding to the openqasm name if the name is a valid name
    :rtype: OperationBase class
    """
    mapping = {
        "CX": CNOT,
        "cx": CNOT,
        "x": SigmaX,
        "y": SigmaY,
        "z": SigmaZ,
        "h": Hadamard,
        "s": Phase,
        "p": Phase,
        "cz": CZ,
        "classical x": ClassicalCNOT,
        "classical z": ClassicalCZ,
        "classical reset x": MeasurementCNOTandReset,
    }
    if name in mapping:
        return mapping[name]
    return None


def class_to_name_mapping(class_op):
    """
    Function to map operation class to string. It's used to convert circuit object to json.

    :param class_op: operation class
    :type class_op: operation
    :return:
    """
    mapping = {
        CNOT: "CX",
        SigmaX: "x",
        SigmaY: "y",
        SigmaZ: "z",
        Hadamard: "h",
        Phase: "s",
        CZ: "cz",
        ClassicalCNOT: "classical x",
        ClassicalCZ: "classical z",
        MeasurementCNOTandReset: "measurement-controlled x and reset",
    }
    if class_op in mapping:
        return mapping[class_op]
    return None
