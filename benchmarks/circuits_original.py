"""
Common circuits for testing

"""
import numpy as np
import networkx as nx

from src.circuit import CircuitDAG
from src.ops import *

from src.backends.density_matrix.functions import ketz0_state, ketz1_state, tensor, ket2dm, partial_trace
from src.backends.density_matrix.state import DensityMatrix


def bell_state_circuit():
    """
    Two qubit Bell state preparation circuit
    """
    ideal_state = dict(
        dm=ket2dm((tensor(2 * [ketz0_state()]) + tensor(2 * [ketz1_state()])) / np.sqrt(2)),
    )
    circuit = CircuitDAG(n_emitter=2, n_classical=0)
    circuit.add(Hadamard(register=0))
    circuit.add(CNOT(control=0, control_type='e', target=1, target_type='e'))
    return circuit, ideal_state


def ghz3_state_circuit():
    """
    Three qubit GHZ state, see examples in literature folder
    """
    ideal_state = dict(
        dm=ket2dm((tensor(3 * [ketz0_state()]) + tensor(3 * [ketz1_state()])) / np.sqrt(2)),
    )

    circuit = CircuitDAG(n_emitter=1, n_photon=3, n_classical=1)
    circuit.add(Hadamard(register=0, reg_type='e'))  # reg_type='e' by default for single-qubit gates
    # control_type, target_type not necessary since this is their default value), but added to explain class API
    circuit.add(CNOT(control=0, control_type='e', target=0, target_type='p'))
    circuit.add(CNOT(control=0, control_type='e', target=1, target_type='p'))
    #circuit.add(Hadamard(register=1, reg_type='p'))
    circuit.add(CNOT(control=0, control_type='e', target=2, target_type='p'))
    circuit.add(Hadamard(register=2, reg_type='p'))
    circuit.add(Hadamard(register=0, reg_type='e'))

    circuit.add(MeasurementCNOTandReset(control=0, control_type='e',
                                        target=2, target_type='p',
                                        c_register=0))
    circuit.add(Hadamard(register=2, reg_type='p'))
    return circuit, ideal_state


def ghz4_state_circuit():
    """
    Four qubit GHZ state, see examples in literature folder
    """
    ideal_state = dict(
        dm=ket2dm((tensor(4 * [ketz0_state()]) + tensor(4 * [ketz1_state()])) / np.sqrt(2)),
    )

    circuit = CircuitDAG(n_emitter=1, n_photon=4, n_classical=1)
    circuit.add(Hadamard(register=0, reg_type='e'))
    circuit.add(CNOT(control=0, control_type='e', target=0, target_type='p'))
    circuit.add(CNOT(control=0, control_type='e', target=1, target_type='p'))
    #circuit.add(Hadamard(register=1, reg_type='p'))

    circuit.add(CNOT(control=0, control_type='e', target=2, target_type='p'))
    #circuit.add(Hadamard(register=2, reg_type='p'))

    circuit.add(CNOT(control=0, control_type='e', target=3, target_type='p'))
    circuit.add(Hadamard(register=3, reg_type='p'))
    circuit.add(Hadamard(register=0, reg_type='e'))

    circuit.add(MeasurementCNOTandReset(control=0, control_type='e',
                                        target=3, target_type='p',
                                        c_register=0))
    circuit.add(Hadamard(register=3, reg_type='p'))
    return circuit, ideal_state


def linear_cluster_3qubit_circuit():
    """
    Three qubit linear cluster state, see examples in literature folder
    """

    graph = nx.Graph([(1, 2), (2, 3)])
    state = DensityMatrix.from_graph(graph)
    ideal_state = dict(
        dm=state.data,
    )

    circuit = CircuitDAG(n_emitter=1, n_photon=3, n_classical=1)
    circuit.add(Hadamard(register=0, reg_type='e'))
    circuit.add(CNOT(control=0, control_type='e', target=0, target_type='p'))
    circuit.add(Hadamard(register=0, reg_type='e'))
    circuit.add(CNOT(control=0, control_type='e', target=1, target_type='p'))
    circuit.add(CNOT(control=0, control_type='e', target=2, target_type='p'))
    circuit.add(Hadamard(register=2, reg_type='p'))
    circuit.add(Hadamard(register=0, reg_type='e'))

    circuit.add(MeasurementCNOTandReset(control=0, control_type='e',
                                        target=2, target_type='p',
                                        c_register=0))

    return circuit, ideal_state


def linear_cluster_4qubit_circuit():
    """
    Four qubit linear cluster state, see examples in literature folder
    """

    graph = nx.Graph([(1, 2), (2, 3), (3, 4)])
    state = DensityMatrix.from_graph(graph)
    ideal_state = dict(
        dm=state.data,
    )

    circuit = CircuitDAG(n_emitter=1, n_photon=4, n_classical=1)
    circuit.add(Hadamard(register=0, reg_type='e'))
    circuit.add(CNOT(control=0, control_type='e', target=0, target_type='p'))
    circuit.add(Hadamard(register=0, reg_type='e'))
    circuit.add(CNOT(control=0, control_type='e', target=1, target_type='p'))
    circuit.add(Hadamard(register=0, reg_type='e'))
    circuit.add(CNOT(control=0, control_type='e', target=2, target_type='p'))
    circuit.add(CNOT(control=0, control_type='e', target=3, target_type='p'))

    circuit.add(Hadamard(register=3, reg_type='p'))
    circuit.add(Hadamard(register=0, reg_type='e'))

    circuit.add(MeasurementCNOTandReset(control=0, control_type='e',
                                        target=3, target_type='p',
                                        c_register=0))

    return circuit, ideal_state
