"""
Common circuits for testing

"""
import numpy as np

from src.circuit import CircuitDAG
from src.ops import *

from src.backends.density_matrix.functions import ketz0_state, ketz1_state, tensor, ket2dm, partial_trace


def bell_state_circuit():
    """
    Two qubit Bell state preparation circuit
    """
    ideal_state = dict(
        dm=ket2dm((tensor(2 * [ketz0_state()]) + tensor(2 * [ketz1_state()])) / np.sqrt(2)),
    )
    circuit = CircuitDAG(2, 0)
    circuit.add(Hadamard(register=0))
    circuit.add(CNOT(control=0, target=1))
    return circuit, ideal_state


def ghz3_state_circuit():
    """
    Three qubit GHZ state, see examples in literature folder
    """
    ideal_state = dict(
        dm=ket2dm((tensor(3 * [ketz0_state()]) + tensor(3 * [ketz1_state()])) / np.sqrt(2)),
    )

    circuit = CircuitDAG(4, 1)
    circuit.add(Hadamard(register=3))
    circuit.add(CNOT(control=3, target=0))
    circuit.add(CNOT(control=3, target=1))
    circuit.add(Hadamard(register=1))
    circuit.add(Hadamard(register=1))

    circuit.add(CNOT(control=3, target=2))
    circuit.add(Hadamard(register=2))
    circuit.add(Hadamard(register=3))

    circuit.add(CNOT(control=3, target=2))

    circuit.add(Hadamard(register=2))

    circuit.add(MeasurementZ(register=3, c_register=0))

    return circuit, ideal_state


def ghz4_state_circuit():
    """
    Four qubit GHZ state, see examples in literature folder
    """
    ideal_state = dict(
        dm=ket2dm((tensor(4 * [ketz0_state()]) + tensor(4 * [ketz1_state()])) / np.sqrt(2)),
    )

    circuit = CircuitDAG(5, 1)
    circuit.add(Hadamard(register=4))
    circuit.add(CNOT(control=4, target=0))
    circuit.add(CNOT(control=4, target=1))
    circuit.add(Hadamard(register=1))
    circuit.add(Hadamard(register=1))

    circuit.add(CNOT(control=4, target=2))
    circuit.add(Hadamard(register=2))
    circuit.add(Hadamard(register=2))

    circuit.add(CNOT(control=4, target=3))
    circuit.add(Hadamard(register=3))

    circuit.add(Hadamard(register=4))
    circuit.add(CNOT(control=4, target=3))
    circuit.add(Hadamard(register=3))

    circuit.add(MeasurementZ(register=4, c_register=0))

    return circuit, ideal_state
