"""
Compilation tools for simulating a circuit with a Graph backend
"""

from graphiq.circuit import ops as ops
from graphiq.backends.compiler_base import CompilerBase


class GraphCompiler(CompilerBase):
    name = "graph"
    ops = [  # the accepted operations for a given compiler
        ops.Input,
        ops.Identity,
        ops.Phase,
        ops.PhaseDagger,
        ops.Hadamard,
        ops.SigmaX,
        ops.SigmaY,
        ops.SigmaZ,
        ops.CNOT,
        ops.CZ,
        ops.ClassicalCNOT,
        ops.ClassicalCZ,
        ops.MeasurementZ,
        ops.MeasurementCNOTandReset,
        ops.Output,
    ]

    def __init__(self, *args, **kwargs):
        """
        Create a compiler which acts on a Graph representation

        :return: nothing
        :rtype: None
        """
        super().__init__(*args, **kwargs)
