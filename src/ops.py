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

    def __init__(self, q_registers=tuple(), c_registers=tuple(), **kwargs):
        self.q_registers = q_registers
        self.c_registers = c_registers

    def assign_action_id(self, nums):
        pass


""" Quantum gates """


class Hadamard(OperationBase):
    def __init__(self, register=None, **kwargs):
        super().__init__(q_registers=(register,), **kwargs)
        self.register = register

    def assign_action_id(self, nums):
        self.register = nums[0]

class SigmaX(OperationBase):
    def __init__(self, register=None, **kwargs):
        super().__init__(q_registers=(register,), **kwargs)
        self.register = register

class SigmaX(SingleTargetOp):
    """

class SigmaY(OperationBase):
    def __init__(self, register=None, **kwargs):
        super().__init__(q_registers=(register,), **kwargs)
        self.register = register


class SigmaZ(OperationBase):
    def __init__(self, register=None, **kwargs):
        super().__init__(q_registers=(register,), **kwargs)
        self.register = register

    """

class CNOT(OperationBase):
    def __init__(self, control=None, target=None, **kwargs):
        super().__init__(q_registers=(control, target), **kwargs)
        self.control = control
        self.target = target


class CPhase(OperationBase):
    def __init__(self, control=None, target=None, **kwargs):
        super().__init__(q_registers=(control, target), **kwargs)
        self.control = control
        self.target = target


class ClassicalCNOT(OperationBase):
    def __init__(self, control=None, target=None, c_register=None, **kwargs):
        super().__init__(q_registers=(control, target,), c_registers=(c_register,), **kwargs)
        self.control = control
        self.target = target
        self.c_register = c_register


class ClassicalCPhase(OperationBase):
    def __init__(self, control=None, target=None, c_register=None, **kwargs):
        super().__init__(q_registers=(control, target,), c_registers=(c_register,), **kwargs)
        self.control = control
        self.target = target
        self.c_register = c_register


class MeasurementZ(OperationBase):
    def __init__(self, register=None, c_register=None, **kwargs):
        super().__init__(q_registers=(register,), c_registers=(c_register,), **kwargs)
        self.register = register
        self.c_register = c_register

class SigmaZ(SingleTargetOp):
    """

class Input(OperationBase):
    def __init__(self, register=None, **kwargs):
        super().__init__(q_registers=(register,), **kwargs)
        self.register = register


class Output(OperationBase):
    def __init__(self, register=None, **kwargs):
        super().__init__(q_registers=(register,), **kwargs)
        self.register = register
