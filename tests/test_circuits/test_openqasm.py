import pytest
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit

import src.ops as ops
import src.visualizers.openqasm.openqasm_lib as oq_lib
from src.circuit import CircuitDAG
from tests.test_flags import visualization

OPENQASM_V = 2


@pytest.fixture(scope='function')
def all_gate_circuit():
    dag = CircuitDAG()
    dag.add_emitter_register(size=1)
    dag.add_photonic_register(size=1)
    dag.add_classical_register(size=1)

    # single qubit gates
    dag.add(ops.SigmaX(register=0, reg_type='e'))
    dag.add(ops.SigmaY(register=0, reg_type='e'))
    dag.add(ops.SigmaZ(register=0, reg_type='p'))
    dag.add(ops.Hadamard(register=0, reg_type='e'))
    dag.add(ops.Phase(register=0, reg_type='p'))

    # controlled gates
    dag.add(ops.CNOT(control=0, control_type='e', target=0, target_type='p'))
    dag.add(ops.CPhase(control=0, control_type='e', target=0, target_type='p'))

    # Controlled measurements
    dag.add(ops.ClassicalCNOT(control=0, control_type='e', target=0, target_type='p', c_register=0))
    dag.add(ops.ClassicalCPhase(control=0, control_type='e', target=0, target_type='p', c_register=0))

    # Measurement
    dag.add(ops.MeasurementZ(register=0, reg_type='p', c_register=0))

    return dag


def check_openqasm_equivalency(s1, s2):
    assert "".join(s1.split()) == "".join(s2.split()), f"Strings don't match. S1 is: \n{''.join(s1.split())}, \n\n, S2 is: \n{''.join(s2.split())}"
    # we remove all white spaces, since openqasm does not consider white spaces


def test_empty_circuit_1(dag):
    openqasm = dag.to_openqasm()
    check_openqasm_equivalency(openqasm, oq_lib.openqasm_header())


def test_gateless_circuit_1(dag):
    dag.add_emitter_register(size=1)
    dag.add_emitter_register(size=1)
    dag.add_classical_register(size=1)
    openqasm = dag.to_openqasm()
    expected = oq_lib.openqasm_header() + oq_lib.register_initialization_string([1, 1], [], [1])
    check_openqasm_equivalency(openqasm, expected)


def test_all_gates_1(all_gate_circuit):
    all_gate_circuit.validate()
    try:
        qasm_str = all_gate_circuit.to_openqasm()
        QuantumCircuit.from_qasm_str(qasm_str)
    except Exception as e:
        print(qasm_str)
        raise e


def test_wrapper_gate_1(all_gate_circuit):
    pass


@visualization
def test_visualization_1(dag):
    # Empty openQASM
    dag.draw_circuit()


@visualization
def test_visualization_2(dag):
    # no operations, but some registers
    dag.add_emitter_register()
    dag.add_emitter_register()
    dag.add_photonic_register()
    dag.add_classical_register()

    # Add CNOT operations
    dag.add(ops.CNOT(control=1, control_type='e', target=0, target_type='p'))

    # Add unitary gates
    dag.add(ops.SigmaX(register=0, reg_type='e'))
    dag.add(ops.Hadamard(register=1, reg_type='p'))
    dag.validate()
    dag.draw_circuit()


@visualization
def test_visualization_3(dag):
    # Create a dag with every gate once
    dag.add_emitter_register(1)
    dag.add_emitter_register(1)
    dag.add_classical_register(1)

    dag.add(ops.Hadamard(register=0))
    dag.add(ops.SigmaX(register=0))
    dag.add(ops.SigmaY(register=0))
    dag.add(ops.SigmaZ(register=0))
    dag.add(ops.CNOT(control=0, control_type='e', target=1, target_type='e'))
    dag.add(ops.CPhase(control=1, control_type='e', target=0, target_type='e'))
    dag.add(ops.MeasurementZ(register=0, reg_type='e', c_register=0))
    fig, ax = dag.draw_circuit(show=False)
    fig.suptitle("testing fig reception")
    plt.show()


@visualization
def test_visualization_all(all_gate_circuit):
    all_gate_circuit.validate()
    try:
        all_gate_circuit.draw_circuit()
    except Exception as e:
        print(all_gate_circuit.to_openqasm())
        raise e
