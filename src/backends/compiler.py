"""
Compiler which takes a circuit description and implements the mapping, given a specific representation of the
underlying quantum state
"""
from abc import ABC, abstractmethod
import warnings
import logging
from functools import reduce

import numpy as np

import src.ops as ops
from src.circuit import Circuit, CircuitDAG

__all__ = [

]

class CompilerBase(ABC):
    """
    Compiles a circuit using a specific representation for the underlying quantum state

    """

    def __init__(self, *args, **kwargs):
        self.name = "base"
        self.ops = [  # the accepted operations for a given compiler
            ops.Input,
            ops.Output,
            ops.OperationBase
        ]

    @abstractmethod
    def compile(self, circuit: CircuitDAG):
        raise NotImplementedError("Please select a valid compiler")

    def check(self, circuit):
        seq = circuit.sequence()

        for i, op in enumerate(seq):
            if type(op) in self.ops:
                logging.info(f"Operation {i} {type(op).__name__} is valid with {type(self).__name__}")
            else:
                logging.error(f"Error: Operation {i} {type(op).__name__} is not valid with {type(self).__name__}")


class DensityMatrixCompiler(CompilerBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "density_matrix"
        self.ops = [  # the accepted operations for a given compiler
            ops.Input,
            ops.CNOT,
            ops.Hadamard,
            ops.PauliX,
            ops.Output,
        ]

    def compile(self, circuit: CircuitDAG):

        sources = [x for x in circuit.dag.nodes() if circuit.dag.in_degree(x) == 0]
        # state = circuit.initial_state()  # TODO: how to get the initial state?

        init = np.outer(np.array([1, 1]/np.sqrt(2)), np.array([1, 1])/np.sqrt(2)).astype('complex64')
        # init = np.outer(np.array([1, 0]), np.array([1, 0])).astype('complex64')
        state = reduce(np.kron, 4*[init])
        print(state)
        seq = circuit.sequence()
        for op in seq:
            if type(op) is ops.OperationBase:  # TODO:
                print(op)
