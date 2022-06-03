"""
The Operation objects are objects which tell the compiler what gate to apply on which registers / qubits

They serve two purposes:
1. They are used to build our circuit: circuit.add(OperationObj). The circuit constructs the DAG structure from the
connectivity information in OperationObj
2. They are passed as an iterable to the compiler, such that the compiler can perform the correct series of information

REGISTERS VS QUDITS/CBITS: following qiskit/openQASM and other softwares, we allow a distinction between registers
and qudits/qubits (where one register can contain a number of qubits / qudits).

IF an operation is given q_registers=(a, b), the circuit takes it to apply between registers a and b (that is,
it applies between all qubits a[j], b[j] for 0 < j < n with a, b having length n)

If an operation is given q_registers=((a, b), (c, d)), the circuit takes it to apply between qubit b of register a,
and qubit d of register c.

We can also use a mixture of registers an qubits: q_registers=(a, (b, c)) means that the operation will be applied
between EACH QUBIT of register a, and qubit c of register b

TODO: consider refactoring register notation to not use tuples (which can be confusing).
"""
from abc import ABC
import src.visualizers.openqasm.openqasm_lib as oq_lib


""" Base classes from which operations will inherit """


class OperationBase(ABC):
    """
    """
    _openqasm_info = None  # This is the information necessary to add our Operation into an openQASM script
    # If _openqasm_info is None, a given operation cannot be added to openQASM

    def __init__(self, q_registers=tuple(), c_registers=tuple(), *args, **kwargs):
        """
        Creates an Operation base object (which is largely responsible for holding the registers on which
        an operation act--this provides a consistent API for the circuit class to use when dealing with
        any arbitrary operation).

        :param q_registers: (a, ..., b) indices (indicating registers a, ..., b) OR
                            ((a, b), ...) indicating qubit b of register a OR
                            any combination of (reg, bit) notation and reg notation
                            These tuples can be of length 1 or empty as well, depending on the number
                            of registers the gate requires
        :type q_registers: tuple (tuple of: integers OR tuples of length 2)
        :param c_registers: same as q_registers, but for the classical registers
        :type c_registers: tuple (tuple of: integers OR tuples of length 2)
        :raises AssertionError: if q_registers, c_registers are not tuples, OR if the elements of the tuple
                               do not correspond to the notation described above
        :return: the function returns nothing
        :rtype: None
        """
        assert isinstance(q_registers, tuple)
        assert isinstance(c_registers, tuple)

        for q in q_registers:
            assert isinstance(q, int) or \
                   (isinstance(q, tuple) and len(q) == 2 and isinstance(q[0], int) and isinstance(q[1], int)), \
                   f'Invalid q_register: q_register tuple must only contain tuples of length 2 or integers'
        for c in c_registers:
            assert isinstance(c, int) or \
                   (isinstance(c, tuple) and len(c) == 2 and isinstance(c[0], int) and isinstance(c[1], int)), \
                   f'Invalid c_register: c_register tuple must only contain tuples of length 2 or integers'

        self._q_registers = q_registers
        self._c_registers = c_registers

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
        Returns the q_registers tuple of the Operation class

        :return: the q_registers_tuple
        :rtype: tuple
        """
        return self._q_registers

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
        Allows us to change the q_registers object on which an Operation acts
        This should only be used by the circuit class!
        Subclass-specific implementations of this function also update other
        register-related fields (e.g. register, target, control) automatically
        when q_registers is updated

        :param q_reg: new q_register which the operation should target
        :type q_reg: tuple
        :raises ValueError: if the new q_reg object does not match the length of self.q_registers (Operations
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

    def _update_q_reg(self, q_reg):
        """
        Helper function to verify the validity of the new q_registers tuple and to update the fields
        This is broken into a separate function because subclasses will need to use this as well,
        and using the super() keyword gets messy with class properties

        :param q_reg: new q_register which the operation should target
        :type q_reg: tuple
        :raises ValueError: if the new q_reg object does not match the length of self.q_registers (Operations
                           should not have variable register numbers)
        :return: function returns nothing
        :rtype: None
        """
        if len(q_reg) != len(self._q_registers):
            raise ValueError(f'The number of quantum registers on which the operation acts cannot be changed!')
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
            raise ValueError(f'The number of classical registers on which the operation acts cannot be changed!')
        self._c_registers = c_reg


class SingleQubitOperationBase(OperationBase):
    """
    This is used as a base class for any single-qubit operation (single-qubit operations should
    all depend on a single parameter, "register"
    """
    def __init__(self, register=None, *args, **kwargs):
        """
        Creates a single-qubit operation base class object

        :param register: the (quantum) register on which the single-qubit operation acts
        :type register: int OR tuple (of ints, length 2)
        :return: function returns nothing
        :rtype: None
        """
        super().__init__(q_registers=(register,), *args, **kwargs)
        self.register = register

    @OperationBase.q_registers.setter
    def q_registers(self, q_reg):
        """
        Handle to modify the register-qubit pairs on which the operation acts. This also automatically updates the
        self.register field

        :param q_reg: the new q_register value to set
        :raises ValueError: if the new q_reg object does not have a length of 1
        :return: function returns nothing
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

    def __init__(self, register, reg_type='q', *args, **kwargs):
        """
        Creates an IO base class Operation

        :param register: the register/qubit which this I/O operation acts on
        :type register: int OR tuple (of ints, length 2)
        :param reg_type: the input/output is for a quantum register if 'q', and for a classical register if 'c'
        :type reg_type: str
        """
        if reg_type == 'q':
            super().__init__(q_registers=(register, ), *args, **kwargs)
        elif reg_type == 'c':
            super().__init__(c_registers=(register, ), *args, **kwargs)
        else:
            raise ValueError("Register type must be either quantum (reg_type='q') or classical (reg_type='c')")
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
        if self.reg_type == 'q':
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
        if self.reg_type == 'c':
            self.register = c_reg[0]


class ControlledPairOperationBase(OperationBase):
    """
    This is used as a base class for our quantum controlled gates (e.g. CNOT, CPHASE). Each ControlledPairOperationBase
    should have control and target registers/qubits specified
    """
    def __init__(self, control=None, target=None, *args, **kwargs):
        """
        Creates a control gate object

        :param control: control register/qubit for the Operation
        :type control: int OR tuple (of ints, length 2)
        :param target: target register/qubit for the Operation
        :type target: int OR tuple (of ints, length 2)
        :return: function returns nothing
        :rtype: None
        """
        super().__init__(q_registers=(control, target), *args, **kwargs)
        self.control = control
        self.target = target

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
    def __init__(self, control=None, target=None, c_register=None, *args, **kwargs):
        """
        Creates the classically controlled gate

        :param control: the control register/qubit
        :type control: int OR tuple (of ints, length 2)
        :param target: the target register/qubit
        :type control: int OR tuple (of ints, length 2)
        :param c_register: the classical register/cbit
        :type c_register: int OR tuple (of ints, length 2)
        :return: the function returns nothing
        :rtype: None
        """
        super().__init__(q_registers=(control, target,), c_registers=(c_register,), *args, **kwargs)
        self.control = control
        self.target = target
        self.c_register = c_register

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


class Hadamard(SingleQubitOperationBase):
    """
    Hadamard gate Operation
    """
    _openqasm_info = oq_lib.hadamard_info()

    def __init__(self, register=None, **kwargs):
        super().__init__(register=register, **kwargs)


class SigmaX(SingleQubitOperationBase):
    """
    Pauli X gate Operation
    """
    _openqasm_info = oq_lib.sigma_x_info()

    def __init__(self, register=None, **kwargs):
        super().__init__(register=register, **kwargs)


class SigmaY(SingleQubitOperationBase):
    """
    Pauli Y gate Operation
    """
    _openqasm_info = oq_lib.sigma_y_info()

    def __init__(self, register=None, **kwargs):
        super().__init__(register=register, **kwargs)


class SigmaZ(SingleQubitOperationBase):
    """
    Pauli Z gate Operation
    """
    _openqasm_info = oq_lib.sigma_z_info()

    def __init__(self, register=None, **kwargs):
        super().__init__(register=register, **kwargs)


class CNOT(ControlledPairOperationBase):
    """
    CNOT gate Operation
    """
    _openqasm_info = oq_lib.cnot_info()


class CPhase(ControlledPairOperationBase):
    """
    CPHASE gate Operation
    """
    _openqasm_info = oq_lib.cphase_info()


class ClassicalCNOT(ClassicalControlledPairOperationBase):
    """
    Classical CNOT gate Operation
    """
    # No easy openQASM 2 representation exists, since it's classically controlled
    # _openqasm_info = oq_lib.classical_cnot_info()


class ClassicalCPhase(ClassicalControlledPairOperationBase):
    """
    Classical CPHASE gate Operation
    """
    # No easy openQASM 2 representation exists, since it's classically controlled
    # _openqasm_info = oq_lib.classical_cphase_info()


class MeasurementZ(OperationBase):
    """
    Z Measurement Operation
    TODO: maybe create a base class for measurements in the future IFF we also want to support other measurements
    """
    _openqasm_info = oq_lib.z_measurement_info()

    def __init__(self, register=None, c_register=None, **kwargs):
        """
        Creates a Z measurement Operation

        :param register: the quantum register on which the measurement is performed
        :type register: int OR tuple (of ints, length 2)
        :param c_register: the classical register to which the measurement result is saved
        :type c_register: int OR tuple (of ints, length 2)
        :return: this function returns nothing
        :rtype: None
        """
        super().__init__(q_registers=(register,), c_registers=(c_register,), **kwargs)
        self.register = register
        self.c_register = c_register

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
    def __init__(self, register=None, reg_type='q', *args, **kwargs):
        super().__init__(register, reg_type=reg_type, *args, **kwargs)


class Output(InputOutputOperationBase):
    """
    Input Operation. Serves as a placeholder in the circuit so that we know that this is the final operation on a
    qubit/cbit (i.e. there are no subsequent operations on it)
    """
    def __init__(self, register=None, reg_type='q', *args, **kwargs):
        super().__init__(register, reg_type=reg_type, *args, **kwargs)
