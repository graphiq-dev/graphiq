"""
Compiler which takes a circuit description and implements the mapping, given a specific representation of the
underlying quantum state
"""
from abc import ABC, abstractmethod
import warnings
import logging
from functools import reduce
import copy

import numpy as np

import src.ops as ops
from src.circuit import CircuitDAG


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
        print(sources)
        # state = circuit.initial_state()  # TODO: how to get the initial state?

        # TODO: make this more general, but for now we assume all registers are initialized to |0>
        init = np.outer(np.array([1, 0]), np.array([1, 0])).astype('complex64')  # initialization of quantum registers
        state = reduce(np.kron, len(sources)*[init])  # generates the tensor product input density matrix
        print(state)

        seq = circuit.sequence()
        for op in seq:
            # print(op)
            # print(op.register)

            if type(op) not in self.ops:
                raise RuntimeError(f"The {op.__class__.__name__} is not valid with "
                                   f"the {self.__class__.__name__} compiler")

            if type(op) is ops.Input:
                pass

            elif type(op) is ops.Hadamard:
                q = op.register
                h = np.array([[1, 1], [1, -1]]).astype("complex64") / np.sqrt(2)
                us = (q - 1) * [np.identity(2)] + [h] + (circuit.n_quantum - q - 1) * [np.identity(2)]
                u = reduce(np.kron, us)
                state = u @ state @ np.conjugate(u).T

            elif type(op) is ops.PauliX:
                q = op.register
                sx = np.array([[0, 1], [1, 0]]).astype("complex64")
                us = (q - 1) * [np.identity(2)] + [sx] + (circuit.n_quantum - q - 1) * [np.identity(2)]
                u = reduce(np.kron, us)
                state = u @ state @ np.conjugate(u).T

            elif type(op) is ops.CNOT:
                c, t = op.control, op.target
                c0 = np.array([[1, 0], [0, 0]])
                c1 = np.array([[0, 0], [0, 1]])

                sx = np.array([[0, 1], [1, 0]]).astype("complex64")
                us = circuit.n_quantum * [np.identity(2)]

                us0 = copy.deepcopy(us)
                us0[c] = np.array([[1, 0], [0, 0]])

                us1 = copy.deepcopy(us)
                us1[c] = np.array([[0, 0], [0, 1]])
                us1[t] = np.array([[0, 1], [1, 0]])

                u = reduce(np.kron, us0) + reduce(np.kron, us1)
                print(u)

        print(state)