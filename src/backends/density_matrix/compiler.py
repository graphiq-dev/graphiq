from functools import reduce

import numpy as np

import src.backends.density_matrix.functions as dm
from src import ops as ops
from src.compiler import CompilerBase
from src.states import DensityMatrix
from src.circuit import CircuitDAG


def reg_to_index_func(reg_list):
    """
    Returns function which map a (reg, bit) tuple pair into an index between 0 and N - 1
    :param reg: a list, where reg[i] = <# of qudits/cbits in register i>
    :return: A function which maps an input tuple (reg, bit) to an index between
             0 and N - 1, where N is the total number of elements across all registers
             (i.e. is the sum of reg)
    """
    reg_array = np.array(reg_list)
    cumulative_num_reg = np.cumsum(reg_array)

    def reg_to_index(reg):
        assert isinstance(reg, tuple) and len(reg) == 2, f'Register must be provided as a tuple of length 2'
        if reg[0] == 0:
            return reg[1]
        else:
            return cumulative_num_reg[reg[0] - 1] + reg[1]

    return reg_to_index


class DensityMatrixCompiler(CompilerBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "density_matrix"
        self.ops = [  # the accepted operations for a given compiler
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

    def compile(self, circuit: CircuitDAG):
        # TODO: using just the source nodes doesn't distinguish classical and quantum
        # sources = [x for x in circuit.dag.nodes() if circuit.dag.in_degree(x) == 0]

        # TODO: make this more general, but for now we assume all registers are initialized to |0>
        init = np.outer(np.array([1, 0]), np.array([1, 0])).astype('complex64')  # initialization of quantum registers

        # TODO: state_id? what should it be? There should be a default.
        state = DensityMatrix(data=reduce(np.kron, circuit.n_quantum * [init]), state_id=0)
        classical_registers = np.zeros(circuit.n_classical)

        # TODO: support self-defined mapping functions later instead of using the default above?
        q_index = reg_to_index_func(circuit.q_registers)
        c_index = reg_to_index_func(circuit.c_registers)

        seq = circuit.sequence()
        for op in seq:
            if type(op) not in self.ops:
                raise RuntimeError(f"The {op.__class__.__name__} is not valid with "
                                   f"the {self.__class__.__name__} compiler")

            if type(op) is ops.Input:
                pass  # TODO: should think about best way to handle inputs/outputs

            elif type(op) is ops.Output:
                pass

            elif type(op) is ops.Hadamard:
                unitary = dm.get_single_qubit_gate(circuit.n_quantum, q_index(op.register), dm.hadamard())
                state.apply_unitary(unitary)

            elif type(op) is ops.SigmaX:
                unitary = dm.get_single_qubit_gate(circuit.n_quantum, q_index(op.register), dm.sigmax())
                state.apply_unitary(unitary)

            elif type(op) is ops.SigmaY:
                unitary = dm.get_single_qubit_gate(circuit.n_quantum, q_index(op.register), dm.sigmay())
                state.apply_unitary(unitary)

            elif type(op) is ops.SigmaZ:
                unitary = dm.get_single_qubit_gate(circuit.n_quantum, q_index(op.register), dm.sigmaz())
                state.apply_unitary(unitary)

            elif type(op) is ops.CNOT:
                unitary = dm.get_controlled_gate(circuit.n_quantum, q_index(op.control),
                                                 q_index(op.target), dm.sigmax())
                state.apply_unitary(unitary)

            elif type(op) is ops.CPhase:
                unitary = dm.get_controlled_gate(circuit.n_quantum, q_index(op.control),
                                                 q_index(op.target), dm.sigmaz())
                state.apply_unitary(unitary)

            elif type(op) is ops.ClassicalCNOT:
                # TODO: handle conditioned vs unconditioned density operators on the measurement outcome
                projectors = dm.projectors_zbasis(circuit.n_quantum, q_index(op.control))
                outcome = state.apply_measurement(projectors)
                if outcome == 1:  # condition an X gate on the target qubit based on the measurement outcome
                    unitary = dm.get_single_qubit_gate(circuit.n_quantum, q_index(op.target), dm.sigmax())
                    state.apply_unitary(unitary)

            elif type(op) is ops.ClassicalCPhase:
                # TODO: handle conditioned vs unconditioned density operators on the measurement outcome
                projectors = dm.projectors_zbasis(circuit.n_quantum, q_index(op.control))
                outcome = state.apply_measurement(projectors)
                if outcome == 1:  # condition a Z gate on the target qubit based on the measurement outcome
                    unitary = dm.get_single_qubit_gate(circuit.n_quantum, q_index(op.target), dm.sigmaz())
                    state.apply_unitary(unitary)

            elif type(op) is ops.MeasurementZ:
                # TODO: handle conditioned vs unconditioned density operators on the measurement outcome
                projectors = dm.projectors_zbasis(circuit.n_quantum, q_index(op.register))
                outcome = state.apply_measurement(projectors)
                classical_registers[c_index(op.c_register)] = outcome

            elif type(op) is ops.Output:
                pass  # TODO: should there be more interaction with input/output nodes?

            else:
                raise RuntimeError(f"{type(op)} is invalid or not implemented for {self.__class__.__name__}.")

        return state
