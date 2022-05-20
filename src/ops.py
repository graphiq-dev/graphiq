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
"""
from abc import ABC


""" Base classes from which operations will inherit """


class OperationBase(ABC):
    """
    """
    def __init__(self, q_registers=tuple(), c_registers=tuple()):
        """
        We assume that tuples refer only to single-qubit registers, by default
        :param q_registers: (a, b, c) indexes
        :param c_registers:
        """
        for q in q_registers:
            assert isinstance(q, int) or \
                   (isinstance(q, tuple) and len(q) == 2 and isinstance(q[0], int) and isinstance(q[1], int)), \
                   f'Invalid q_register: q_register tuple must only contain tuples of length 2 or integers'
        for c in c_registers:
            assert isinstance(c, int) or \
                   (isinstance(c, tuple) and len(c) == 2 and isinstance(c[0], int) and isinstance(c[1], int)), \
                   f'Invalid c_register: c_register tuple must only contain tuples of length 2 or integers'

        self.q_registers = q_registers
        self.c_registers = c_registers

    def update_register_indices(self, q_registers, c_registers):
        pass


class SingleQubitOperationBase(OperationBase):
    def __init__(self, register=None, *args, **kwargs):
        super().__init__(q_registers=(register,), *args, **kwargs)
        self.register = register

    def update_register_indices(self, q_registers, c_registers):
        self.register = q_registers[0]
        self.q_registers = q_registers
        self.c_registers = c_registers


class InputOutputOperationBase(OperationBase):
    def __init__(self, register, reg_type='q', *args, **kwargs):
        if reg_type == 'q':
            super().__init__(q_registers=(register, ), *args, **kwargs)
        else:
            super().__init__(c_registers=(register, ), *args, **kwargs)
        self.register = register

    def update_register_indices(self, q_registers, c_registers):
        self.register = q_registers[0]
        self.q_registers = q_registers
        self.c_registers = c_registers


""" Quantum gates """


class Hadamard(SingleQubitOperationBase):
    def __init__(self, register=None, **kwargs):
        super().__init__(q_registers=(register,), **kwargs)
        self.register = register


class SigmaX(SingleQubitOperationBase):
    def __init__(self, register=None, **kwargs):
        super().__init__(q_registers=(register,), **kwargs)
        self.register = register


class SigmaY(SingleQubitOperationBase):
    def __init__(self, register=None, **kwargs):
        super().__init__(q_registers=(register,), **kwargs)
        self.register = register


class SigmaZ(SingleQubitOperationBase):
    def __init__(self, register=None, **kwargs):
        super().__init__(q_registers=(register,), **kwargs)
        self.register = register


class CNOT(OperationBase):
    def __init__(self, control=None, target=None, **kwargs):
        super().__init__(q_registers=(control, target), **kwargs)
        self.control = control
        self.target = target

    def update_register_indices(self, q_registers, c_registers):
        self.control = q_registers[0]
        self.target = q_registers[1]
        self.q_registers = q_registers
        self.c_registers = c_registers


class CPhase(OperationBase):
    def __init__(self, control=None, target=None, **kwargs):
        super().__init__(q_registers=(control, target), **kwargs)
        self.control = control
        self.target = target

    def update_register_indices(self, q_registers, c_registers):
        self.control = q_registers[0]
        self.target = q_registers[1]
        self.q_registers = q_registers
        self.c_registers = c_registers


class ClassicalCNOT(OperationBase):
    def __init__(self, control=None, target=None, c_register=None, **kwargs):
        super().__init__(q_registers=(control, target,), c_registers=(c_register,), **kwargs)
        self.control = control
        self.target = target
        self.c_register = c_register

    def update_register_indices(self, q_registers, c_registers):
        self.control = q_registers[0]
        self.target = q_registers[1]
        self.c_register = c_registers[0]

        self.q_registers = q_registers
        self.c_registers = c_registers


class ClassicalCPhase(OperationBase):
    def __init__(self, control=None, target=None, c_register=None, **kwargs):
        super().__init__(q_registers=(control, target,), c_registers=(c_register,), **kwargs)
        self.control = control
        self.target = target
        self.c_register = c_register

    def update_register_indices(self, q_registers, c_registers):
        self.control = q_registers[0]
        self.target = q_registers[1]
        self.c_register = c_registers[0]

        self.q_registers = q_registers
        self.c_registers = c_registers


class MeasurementZ(OperationBase):
    def __init__(self, register=None, c_register=None, **kwargs):
        super().__init__(q_registers=(register,), c_registers=(c_register,), **kwargs)
        self.register = register
        self.c_register = c_register

    def update_register_indices(self, q_registers, c_registers):
        self.register = q_registers[0]
        self.c_register = c_registers[0]

        self.q_registers = q_registers
        self.c_registers = c_registers


class Input(InputOutputOperationBase):
    def __init__(self, register=None, **kwargs):
        super().__init__(q_registers=(register,), **kwargs)
        self.register = register

    # TODO: update_register_indices here??


class Output(InputOutputOperationBase):
    def __init__(self, register=None, **kwargs):
        super().__init__(q_registers=(register,), **kwargs)
        self.register = register
    # TODO: update_register_indices here??
