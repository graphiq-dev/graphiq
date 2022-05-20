"""

"""
from abc import ABC


class OperationBase(ABC):
    """

    """
    def __init__(self, q_registers=tuple(), c_registers=tuple(), **kwargs):
        self.q_registers = q_registers
        self.c_registers = c_registers


""" Quantum gates """


class Hadamard(OperationBase):
    def __init__(self, register=None, **kwargs):
        super().__init__(q_registers=(register,), **kwargs)
        self.register = register


class SigmaX(OperationBase):
    def __init__(self, register=None, **kwargs):
        super().__init__(q_registers=(register,), **kwargs)
        self.register = register


class SigmaY(OperationBase):
    def __init__(self, register=None, **kwargs):
        super().__init__(q_registers=(register,), **kwargs)
        self.register = register


class SigmaZ(OperationBase):
    def __init__(self, register=None, **kwargs):
        super().__init__(q_registers=(register,), **kwargs)
        self.register = register


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


class Input(OperationBase):
    def __init__(self, register=None, **kwargs):
        super().__init__(q_registers=(register,), **kwargs)
        self.register = register


class Output(OperationBase):
    def __init__(self, register=None, **kwargs):
        super().__init__(q_registers=(register,), **kwargs)
        self.register = register
