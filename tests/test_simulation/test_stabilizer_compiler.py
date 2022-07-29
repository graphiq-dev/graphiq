from benchmarks.circuits import *
from src.backends.stabilizer.state import Stabilizer
from src.backends.stabilizer.tableau import CliffordTableau
from src.backends.stabilizer.compiler import StabilizerCompiler
import src.backends.state_representation_conversion as converter
import numpy as np


def test_linear_3qubit():
    n_qubits = 4
    circuit, state = linear_cluster_3qubit_circuit()
    compiler = StabilizerCompiler()
    output_stabilizer = compiler.compile(circuit)
    print(output_stabilizer.tableau)
    # TODO: validate the result


def test_linear_4qubit():
    n_qubits = 5
    circuit, state = linear_cluster_4qubit_circuit()
    compiler = StabilizerCompiler()
    output_stabilizer = compiler.compile(circuit)
    print(output_stabilizer.tableau)
    # TODO: validate the result
