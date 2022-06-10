"""
Compilation tools for simulating a circuit with a purely Density Matrix based backend
"""

from functools import reduce

import numpy as np

import src.backends.density_matrix.functions as dm
from src import ops as ops
from src.backends.compiler_base import CompilerBase
from src.backends.density_matrix.state import DensityMatrix
from src.circuit import CircuitBase

# TODO: this is deprecated, as we now only use integers to index the quantum registers (rather than tuples)
# def reg_to_index_func(reg_list):
#     """
#     Returns function which map a (reg, bit) tuple pair into an index between 0 and N - 1
#     This allows the compiler to correctly assign a unique index to each qubit/cbit in each register
#
#     :param reg_list: a list, where reg[i] = <# of qudits/cbits in register i>
#     :type reg_list: list (of ints)
#     :return: A function which maps an input tuple (reg, bit) to an index between
#              0 and N - 1, where N is the total number of elements across all registers
#              (i.e. is the sum of reg)
#     :rtype: function
#     """
#     reg_array = np.array(reg_list)
#     cumulative_num_reg = np.cumsum(reg_array)
#
#     def reg_to_index(reg):
#         # TODO: THIS IS DEPRECATED
#         """ Function which maps (reg, bit) to a unique index """
#         assert isinstance(reg, tuple) and len(reg) == 2, f'Register must be provided as a tuple of length 2'
#         if reg[0] == 0:
#             return reg[1]
#         else:
#             return cumulative_num_reg[reg[0] - 1] + reg[1]
#
#     return reg_to_index


def reg_to_index_func(n_photon):
    def reg_to_index(reg, reg_type):
        if reg_type == 'p':
            return reg
        elif reg_type == 'e':
            return reg + n_photon

    return reg_to_index

class DensityMatrixCompiler(CompilerBase):
    """
    Compiler which deals exclusively with the DensityMatrix state representation.
    Currently creates a DensityMatrix state object and applies the circuit Operations to it in order

    # TODO: refactor to return a QuantumState object rather than a DensityMatrix object
    # TODO: [longer term] refactor to take a QuantumState object input instead of creating its own initial state?
    """
    name = "density_matrix"
    ops = [  # the accepted operations for a given compiler
        ops.Input,
        ops.Hadamard,
        ops.SigmaX,
        ops.SigmaY,
        ops.SigmaZ,
        ops.CNOT,
        ops.CPhase,
        ops.ClassicalCNOT,
        ops.ClassicalCPhase,
        ops.MeasurementZ,
        ops.Output,
    ]
    def __init__(self, *args, **kwargs):
        """
        Create a compiler which acts on a DensityMatrix state representation

        :return: function returns nothing
        :rtype: None
        """
        super().__init__(*args, **kwargs)

    def compile(self, circuit: CircuitBase):
        """
        Compiles (i.e. produces an output state) circuit, in density matrix representation.
        This involves sequentially applying each operation of the circuit on the initial state

        :param circuit: the circuit to compile
        :type circuit: CircuitBase
        :raises ValueError: if there is a circuit Operation which is incompatible with this compiler
        :return: the state produced by the circuit
        :rtype: DensityMatrix

        TODO: return a QuantumState object instead
        """
        # TODO: using just the source nodes doesn't distinguish classical and quantum
        # sources = [x for x in circuit.dag.nodes() if circuit.dag.in_degree(x) == 0]

        # TODO: make this more general, but for now we assume all registers are initialized to |0>
        init = np.outer(np.array([1, 0]), np.array([1, 0])).astype('complex64')  # initialization of quantum registers

        # TODO: state_id? what should it be? There should be a default.

        # TODO: refactor to be a QuantumState object which contains a density matrix
        state = DensityMatrix(data=reduce(np.kron, circuit.n_quantum * [init]))
        classical_registers = np.zeros(circuit.n_classical)

        # TODO: support self-defined mapping functions later instead of using the default above
        # Get functions which will map from registers to a unique index
        q_index = reg_to_index_func(circuit.n_photons)

        seq = circuit.sequence()
        for op in seq:
            if type(op) not in self.ops:
                raise RuntimeError(f"The Operation class {op.__class__.__name__} is not valid with "
                                   f"the {self.__class__.__name__} compiler")

            if type(op) is ops.Input:
                pass  # TODO: should think about best way to handle inputs/outputs

            elif type(op) is ops.Output:
                pass

            elif type(op) is ops.Hadamard:
                unitary = dm.get_single_qubit_gate(circuit.n_quantum, q_index(op.register, op.reg_type), dm.hadamard())
                state.apply_unitary(unitary)

            elif type(op) is ops.SigmaX:
                unitary = dm.get_single_qubit_gate(circuit.n_quantum, q_index(op.register, op.reg_type), dm.sigmax())
                state.apply_unitary(unitary)

            elif type(op) is ops.SigmaY:
                unitary = dm.get_single_qubit_gate(circuit.n_quantum, q_index(op.register, op.reg_type), dm.sigmay())
                state.apply_unitary(unitary)

            elif type(op) is ops.SigmaZ:
                unitary = dm.get_single_qubit_gate(circuit.n_quantum, q_index(op.register, op.reg_type), dm.sigmaz())
                state.apply_unitary(unitary)

            elif type(op) is ops.CNOT:
                unitary = dm.get_controlled_gate(circuit.n_quantum, q_index(op.control, op.control_type),
                                                 q_index(op.target, op.target_type), dm.sigmax())
                state.apply_unitary(unitary)

            elif type(op) is ops.CPhase:
                unitary = dm.get_controlled_gate(circuit.n_quantum, q_index(op.control, op.control_type),
                                                 q_index(op.target, op.target_type), dm.sigmaz())
                state.apply_unitary(unitary)

            elif type(op) is ops.ClassicalCNOT:
                # TODO: handle conditioned vs unconditioned density operators on the measurement outcome
                projectors = dm.projectors_zbasis(circuit.n_quantum, q_index(op.control, op.control_type))
                outcome = state.apply_measurement(projectors)
                if outcome == 1:  # condition an X gate on the target qubit based on the measurement outcome
                    unitary = dm.get_single_qubit_gate(circuit.n_quantum,
                                                       q_index(op.target, op.target_type), dm.sigmax())
                    state.apply_unitary(unitary)

            elif type(op) is ops.ClassicalCPhase:
                # TODO: handle conditioned vs unconditioned density operators on the measurement outcome
                projectors = dm.projectors_zbasis(circuit.n_quantum, q_index(op.control, op.control_type))
                outcome = state.apply_measurement(projectors)
                if outcome == 1:  # condition a Z gate on the target qubit based on the measurement outcome
                    unitary = dm.get_single_qubit_gate(circuit.n_quantum,
                                                       q_index(op.target, op.target_type), dm.sigmaz())
                    state.apply_unitary(unitary)

            elif type(op) is ops.MeasurementZ:
                # TODO: handle conditioned vs unconditioned density operators on the measurement outcome
                projectors = dm.projectors_zbasis(circuit.n_quantum, q_index(op.register, op.reg_type))
                outcome = state.apply_measurement(projectors)
                classical_registers[op.c_register] = outcome

            else:
                raise ValueError(f"{type(op)} is invalid or not implemented for {self.__class__.__name__}.")

        return state
