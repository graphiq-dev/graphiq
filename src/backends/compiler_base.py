"""
Compiler which takes a circuit description and implements the mapping, given a specific representation of the
underlying quantum state
"""
from abc import ABC, abstractmethod
import logging

import src.ops as ops
from src.circuit import CircuitBase


class CompilerBase(ABC):
    """
    Compiles a circuit using a specific representation for the underlying quantum state

    """

    def __init__(self, *args, **kwargs):
        self.name = "base"
        self.ops = [  # the accepted operations for a given compiler
            ops.OperationBase
        ]

    @abstractmethod
    def compile(self, circuit: CircuitBase):
        raise NotImplementedError("Please select a valid compiler")

    def check(self, circuit):
        seq = circuit.sequence()

        for i, op in enumerate(seq):
            if type(op) in self.ops:
                logging.info(f"Operation {i} {type(op).__name__} is valid with {type(self).__name__}")
            else:
                logging.error(f"Error: Operation {i} {type(op).__name__} is not valid with {type(self).__name__}")


