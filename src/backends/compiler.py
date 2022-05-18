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
import src.backends.density_matrix_functions as dmf
from src.backends.state_representations import DensityMatrix


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
        # TODO: make this more general, but for now we assume all registers are initialized to |0>
        init = np.outer(np.array([1, 0]), np.array([1, 0])).astype('complex64')  # initialization of quantum registers

        state = DensityMatrix(state_data=reduce(np.kron, len(sources) * [init]), state_id=0)  # TODO: state_id? what should it be? There should be a default.

        seq = circuit.sequence()
        for op in seq:
            if type(op) not in self.ops:
                raise RuntimeError(f"The {op.__class__.__name__} is not valid with "
                                   f"the {self.__class__.__name__} compiler")

            if type(op) is ops.Input:
                pass  # TODO: should think about best way to handle inputs/outputs

            elif type(op) is ops.Hadamard:
                unitary = dmf.get_single_qubit_gate(circuit.n_quantum, op.register, dmf.hadamard())
                state.apply_unitary(unitary)

            elif type(op) is ops.PauliX:
                unitary = dmf.get_single_qubit_gate(circuit.n_quantum, op.register, dmf.sigmax())
                state.apply_unitary(unitary)

            elif type(op) is ops.CNOT:
                unitary = dmf.get_controlled_gate(circuit.n_quantum, op.control, op.target, dmf.sigmax())
                state.apply_unitary(unitary)

        return state
