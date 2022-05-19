from functools import reduce

import numpy as np

import src.backends.density_matrix.functions
from src import ops as ops
from src.backends.compiler_base import CompilerBase
from src.backends.state_representations import DensityMatrix
from src.circuit import CircuitDAG


class DensityMatrixCompiler(CompilerBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "density_matrix"
        self.ops = [  # the accepted operations for a given compiler
            ops.Input,
            ops.CNOT,
            ops.Hadamard,
            ops.SigmaX,
            ops.SigmaY,
            ops.SigmaZ,
            ops.MeasurementZ,
            ops.Output,
        ]

    def compile(self, circuit: CircuitDAG):

        sources = [x for x in circuit.dag.nodes() if circuit.dag.in_degree(x) == 0]
        # TODO: make this more general, but for now we assume all registers are initialized to |0>
        init = np.outer(np.array([1, 0]), np.array([1, 0])).astype('complex64')  # initialization of quantum registers

        # TODO: state_id? what should it be? There should be a default.
        state = DensityMatrix(state_data=reduce(np.kron, len(sources) * [init]), state_id=0)

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
                unitary = src.backends.density_matrix.functions.get_single_qubit_gate(circuit.n_quantum, op.register, src.backends.density_matrix.functions.hadamard())
                state.apply_unitary(unitary)

            elif type(op) is ops.SigmaX:
                unitary = src.backends.density_matrix.functions.get_single_qubit_gate(circuit.n_quantum, op.register, src.backends.density_matrix.functions.sigmax())
                state.apply_unitary(unitary)

            elif type(op) is ops.SigmaY:
                unitary = src.backends.density_matrix.functions.get_single_qubit_gate(circuit.n_quantum, op.register, src.backends.density_matrix.functions.sigmay())
                state.apply_unitary(unitary)

            elif type(op) is ops.SigmaZ:
                unitary = src.backends.density_matrix.functions.get_single_qubit_gate(circuit.n_quantum, op.register, src.backends.density_matrix.functions.sigmaz())
                state.apply_unitary(unitary)

            elif type(op) is ops.CNOT:
                unitary = src.backends.density_matrix.functions.get_controlled_gate(circuit.n_quantum, op.control, op.target, src.backends.density_matrix.functions.sigmax())
                state.apply_unitary(unitary)

            elif type(op) is ops.MeasurementZ:
                # TODO: test the implementation of the measurement
                unitary = src.backends.density_matrix.functions.get_single_qubit_gate(circuit.n_quantum, op.register, src.backends.density_matrix.functions.sigmaz())
                state.apply_unitary(unitary)

            else:
                raise RuntimeError(f"An invalid operation, {type(op)} is contained in the circuit.")

        return state
