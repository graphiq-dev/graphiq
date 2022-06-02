"""
Compilers takes a circuit description and implements the mapping, given a specific representation of the
underlying quantum state.

The Base class defines an API which all compiler implementations should follow
"""
from abc import ABC, abstractmethod
import logging

import src.ops as ops
from src.circuit import CircuitBase


class CompilerBase(ABC):
    """
    Base class for compiler implementations.
    In general, compilers compile circuits using a specific representation(s) for the underlying quantum state
    """
    name = "base"
    ops = [  # the accepted operations for a given compiler
        ops.OperationBase
    ]

    def __init__(self, *args, **kwargs):
        """
        Initializes CompilerBase fields

        :return: function returns nothing
        :rtype: None
        """
        pass

    @abstractmethod
    def compile(self, circuit: CircuitBase):
        raise NotImplementedError("Please select a valid compiler")

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
                logging.info(f"Operation {i} {type(op).__name__} is valid with {type(self).__name__}")
            else:
                logging.error(f"Error: Operation {i} {type(op).__name__} is not valid with {type(self).__name__}")
                valid = False

        return valid
