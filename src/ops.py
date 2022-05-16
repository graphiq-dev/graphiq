"""
"""
from abc import ABC, abstractmethod


class OperationBase(ABC):
    """

    """
    def __init__(self, q_registers=tuple(), c_registers=tuple()):
        self.q_registers = q_registers
        self.c_registers = c_registers


""" Quantum gates """


class CNOT(OperationBase):
    def __init__(self, control=None, target=None, *args, **kwargs):
        super().__init__(q_registers=(control, target), *args, **kwargs)
        self.control = control
        self.target = target


class Hadamard(OperationBase):
    def __init__(self, register=None, *args, **kwargs):
        super().__init__(q_registers=(register,), *args, **kwargs)
        self.register = register


class PauliX(OperationBase):
    def __init__(self, register=None, *args, **kwargs):
        super().__init__(q_registers=(register,), *args, **kwargs)
        self.register = register


class Input(OperationBase):
    def __init__(self, register=None, *args, **kwargs):
        super().__init__(q_registers=(register,), *args, **kwargs)
        self.register = register


class Output(OperationBase):
    def __init__(self, register=None, *args, **kwargs):
        super().__init__(q_registers=(register,), *args, **kwargs)
        self.register = register


