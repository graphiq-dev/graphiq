"""
A suite of circuits for small, known quantum states that can be used for benchmarking, testing, and learning.
"""

import networkx as nx

from graphiq.circuit.circuit_dag import CircuitDAG
from graphiq.circuit.ops import *
from graphiq.backends.density_matrix.functions import (
    state_ketz0,
    state_ketz1,
    tensor,
    ket2dm,
)
from graphiq.backends.density_matrix.state import DensityMatrix
from graphiq.state import QuantumState
from graphiq.backends.stabilizer.clifford_tableau import CliffordTableau
from graphiq.backends.stabilizer.state import Stabilizer
import graphiq.backends.state_rep_conversion as rc
import graphiq.circuit.ops as ops


def bell_state_circuit():
    """
    Two-qubit Bell state preparation circuit
    """
    rho = ket2dm((tensor(2 * [state_ketz0()]) + tensor(2 * [state_ketz1()]))) / 2.0
    ideal_state = QuantumState(rho, rep_type="dm")
    circuit = CircuitDAG(n_emitter=2, n_classical=0)
    circuit.add(Hadamard(register=0))
    circuit.add(CNOT(control=0, control_type="e", target=1, target_type="e"))
    return circuit, ideal_state


def ghz3_state_circuit():
    """
    Three-qubit GHZ state
    """
    rho = ket2dm((tensor(3 * [state_ketz0()]) + tensor(3 * [state_ketz1()]))) / 2.0
    ideal_state = QuantumState(rho, rep_type="dm")
    graph = nx.Graph([(0, 1), (1, 2)])
    c_tableau = CliffordTableau(3)
    s_tableau = rc.graph_to_stabilizer(graph)[0][1]
    c_tableau.stabilizer = s_tableau.table

    ideal_state.stabilizer = Stabilizer(c_tableau)
    ideal_state.stabilizer.apply_hadamard(1)
    ideal_state.stabilizer.apply_hadamard(2)

    circuit = CircuitDAG(n_emitter=1, n_photon=3, n_classical=1)
    circuit.add(Hadamard(register=0, reg_type="e"))
    # reg_type='e' by default for single-qubit gates
    # control_type, target_type not necessary since this is their default value, but added to explain class API
    circuit.add(CNOT(control=0, control_type="e", target=0, target_type="p"))
    circuit.add(CNOT(control=0, control_type="e", target=1, target_type="p"))

    circuit.add(CNOT(control=0, control_type="e", target=2, target_type="p"))
    circuit.add(Hadamard(register=2, reg_type="p"))
    circuit.add(Hadamard(register=0, reg_type="e"))

    circuit.add(
        MeasurementCNOTandReset(
            control=0, control_type="e", target=2, target_type="p", c_register=0
        )
    )
    circuit.add(Hadamard(register=2, reg_type="p"))
    return circuit, ideal_state


def ghz4_state_circuit():
    """
    Four-qubit GHZ state
    """

    rho = ket2dm((tensor(4 * [state_ketz0()]) + tensor(4 * [state_ketz1()]))) / 2.0
    ideal_state = QuantumState(rho, rep_type="dm")
    graph = nx.Graph([(0, 1), (1, 2), (2, 3)])
    c_tableau = CliffordTableau(4)
    s_tableau = rc.graph_to_stabilizer(graph)[0][1]
    c_tableau.stabilizer = s_tableau.table

    ideal_state.stabilizer = Stabilizer(c_tableau)
    ideal_state.stabilizer.apply_hadamard(1)
    ideal_state.stabilizer.apply_hadamard(2)
    ideal_state.stabilizer.apply_hadamard(3)
    circuit = CircuitDAG(n_emitter=1, n_photon=4, n_classical=1)
    circuit.add(Hadamard(register=0, reg_type="e"))
    circuit.add(CNOT(control=0, control_type="e", target=0, target_type="p"))
    circuit.add(CNOT(control=0, control_type="e", target=1, target_type="p"))
    # circuit.add(Hadamard(register=1, reg_type='p'))

    circuit.add(CNOT(control=0, control_type="e", target=2, target_type="p"))
    # circuit.add(Hadamard(register=2, reg_type='p'))

    circuit.add(CNOT(control=0, control_type="e", target=3, target_type="p"))
    circuit.add(Hadamard(register=3, reg_type="p"))
    circuit.add(Hadamard(register=0, reg_type="e"))

    circuit.add(
        MeasurementCNOTandReset(
            control=0, control_type="e", target=3, target_type="p", c_register=0
        )
    )
    circuit.add(Hadamard(register=3, reg_type="p"))
    return circuit, ideal_state


def linear_cluster_3qubit_circuit():
    """
    Three-qubit linear cluster state
    """
    graph = nx.Graph([(1, 2), (2, 3)])
    state = DensityMatrix.from_graph(graph)
    ideal_state = QuantumState(state.data, rep_type="dm")
    c_tableau = CliffordTableau(3)
    s_tableau = rc.graph_to_stabilizer(graph)[0][1]
    c_tableau.stabilizer = s_tableau.table
    ideal_state.stabilizer = Stabilizer(c_tableau)
    circuit = CircuitDAG(n_emitter=1, n_photon=3, n_classical=1)
    circuit.add(Hadamard(register=0, reg_type="e"))
    circuit.add(CNOT(control=0, control_type="e", target=0, target_type="p"))
    circuit.add(Hadamard(register=0, reg_type="e"))
    circuit.add(CNOT(control=0, control_type="e", target=1, target_type="p"))
    circuit.add(CNOT(control=0, control_type="e", target=2, target_type="p"))
    circuit.add(Hadamard(register=2, reg_type="p"))
    circuit.add(Hadamard(register=0, reg_type="e"))

    circuit.add(
        MeasurementCNOTandReset(
            control=0, control_type="e", target=2, target_type="p", c_register=0
        )
    )

    return circuit, ideal_state


def linear_cluster_4qubit_circuit():
    """
    Four-qubit linear cluster state
    """
    graph = nx.Graph([(1, 2), (2, 3), (3, 4)])
    state = DensityMatrix.from_graph(graph)
    ideal_state = QuantumState(state.data, rep_type="dm")
    c_tableau = CliffordTableau(4)
    s_tableau = rc.graph_to_stabilizer(graph)[0][1]
    c_tableau.stabilizer = s_tableau.table
    ideal_state.stabilizer = Stabilizer(c_tableau)
    circuit = CircuitDAG(n_emitter=1, n_photon=4, n_classical=1)
    circuit.add(Hadamard(register=0, reg_type="e"))
    circuit.add(CNOT(control=0, control_type="e", target=0, target_type="p"))
    circuit.add(Hadamard(register=0, reg_type="e"))
    circuit.add(CNOT(control=0, control_type="e", target=1, target_type="p"))
    circuit.add(Hadamard(register=0, reg_type="e"))
    circuit.add(CNOT(control=0, control_type="e", target=2, target_type="p"))
    circuit.add(CNOT(control=0, control_type="e", target=3, target_type="p"))

    circuit.add(Hadamard(register=3, reg_type="p"))
    circuit.add(Hadamard(register=0, reg_type="e"))

    circuit.add(
        MeasurementCNOTandReset(
            control=0, control_type="e", target=3, target_type="p", c_register=0
        )
    )

    return circuit, ideal_state


def variational_entangling_layer_2qubit():
    """
    Variational circuit composed of one local rotation and entangling rotation gate, with the ideal state being a
    Bell state.
    """
    ket = dmf.tensor([dmf.state_ketz0(), dmf.state_ketz0()]) + dmf.tensor(
        [dmf.state_ketz1(), dmf.state_ketz1()]
    )
    state = DensityMatrix(dmf.ket2dm(ket))
    ideal_state = QuantumState(state.data, rep_type="dm")

    circuit = CircuitDAG(n_emitter=2, n_photon=0, n_classical=0)
    circuit.add(ParameterizedOneQubitRotation(register=0, reg_type="e"))
    circuit.add(
        ParameterizedControlledRotationQubit(
            control=0, control_type="e", target=1, target_type="e"
        )
    )

    return circuit, ideal_state


def strongly_entangling_layer(n_qubits=3, layers=1):
    """
    Variational circuit composed of a single entangling layer. Ideal state is an n-qubit GHZ state.
    """

    ket = (
            1
            / np.sqrt(2 ** n_qubits)
            * (
                    dmf.tensor(n_qubits * [dmf.state_ketz0()])
                    + dmf.tensor(n_qubits * [dmf.state_ketz1()])
            )
    )
    state = DensityMatrix(dmf.ket2dm(ket))
    ideal_state = QuantumState(state.data, rep_type="dm")

    circuit = CircuitDAG(n_emitter=n_qubits, n_photon=0, n_classical=0)
    for layer in range(layers):
        for i in range(n_qubits):
            circuit.add(
                ops.ParameterizedOneQubitRotation(
                    register=i,
                    reg_type="e",
                    params=(0.0, 1.0, 1.0),  # set the parameters explicitly, if desired
                )
            )
        for i in range(n_qubits):
            circuit.add(
                ops.CNOT(
                    control=i,
                    control_type="e",
                    target=(i + 1) % n_qubits,
                    target_type="e",
                )
            )

    return circuit, ideal_state
