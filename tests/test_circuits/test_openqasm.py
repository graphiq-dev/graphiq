import pytest
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
import numpy as np
import src.ops as ops
import src.utils.openqasm_lib as oq_lib
from src.circuit import CircuitDAG
from tests.test_flags import visualization

OPENQASM_V = 2


def plot_two_circuits(circuit1, circuit2):
    fig, ax = plt.subplots(2)
    circuit1.draw_circuit(show=False, ax=ax[0])
    circuit2.draw_circuit(show=False, ax=ax[1])
    plt.title("Circuits should be the same!")
    plt.show()


@pytest.fixture(scope="function")
def all_gate_circuit():
    dag = CircuitDAG()
    dag.add_emitter_register(size=1)
    dag.add_emitter_register(size=1)
    dag.add_photonic_register(size=1)
    dag.add_classical_register(size=1)

    # single qubit gates
    dag.add(ops.SigmaX(register=0, reg_type="e"))
    dag.add(ops.SigmaY(register=0, reg_type="e"))
    dag.add(ops.SigmaZ(register=0, reg_type="p"))
    dag.add(ops.Hadamard(register=0, reg_type="e"))
    dag.add(ops.Phase(register=0, reg_type="p"))

    # controlled gates
    dag.add(ops.CNOT(control=0, control_type="e", target=0, target_type="p"))
    dag.add(ops.CZ(control=0, control_type="e", target=0, target_type="p"))

    # Controlled measurements
    dag.add(
        ops.ClassicalCNOT(
            control=0, control_type="e", target=0, target_type="p", c_register=0
        )
    )
    dag.add(
        ops.ClassicalCZ(
            control=0, control_type="e", target=0, target_type="p", c_register=0
        )
    )
    dag.add(
        ops.MeasurementCNOTandReset(
            control=0, control_type="e", target=0, target_type="p", c_register=0
        )
    )

    # Measurement
    dag.add(ops.MeasurementZ(register=0, reg_type="p", c_register=0))

    return dag


@pytest.fixture(scope="function")
def initialization_circuit():
    dag = CircuitDAG(n_emitter=2, n_photon=10, n_classical=1)
    dag.add(ops.CNOT(control=0, control_type="e", target=0, target_type="p"))
    dag.add(ops.CNOT(control=0, control_type="e", target=1, target_type="p"))
    dag.add(ops.CNOT(control=0, control_type="e", target=2, target_type="p"))
    dag.add(ops.CNOT(control=1, control_type="e", target=3, target_type="p"))
    dag.add(ops.CNOT(control=1, control_type="e", target=4, target_type="p"))
    dag.add(ops.CNOT(control=0, control_type="e", target=5, target_type="p"))
    dag.add(ops.CNOT(control=0, control_type="e", target=6, target_type="p"))
    dag.add(ops.CNOT(control=0, control_type="e", target=7, target_type="p"))
    dag.add(ops.CNOT(control=0, control_type="e", target=8, target_type="p"))
    dag.add(ops.CNOT(control=0, control_type="e", target=9, target_type="p"))
    dag.add(
        ops.MeasurementCNOTandReset(
            control=0, control_type="e", target=0, target_type="p", c_register=0
        )
    )
    dag.add(
        ops.MeasurementCNOTandReset(
            control=1, control_type="e", target=4, target_type="p", c_register=0
        )
    )
    return dag


def check_openqasm_equivalency(s1, s2):
    assert "".join(s1.split()) == "".join(
        s2.split()
    ), f"Strings don't match. S1 is: \n{''.join(s1.split())}, \n\n, S2 is: \n{''.join(s2.split())}"
    # we remove all white spaces, since openqasm does not consider white spaces


def test_empty_circuit_1(dag):
    openqasm = dag.to_openqasm()
    check_openqasm_equivalency(openqasm, oq_lib.openqasm_header())


def test_gateless_circuit_1(dag):
    dag.add_emitter_register(size=1)
    dag.add_emitter_register(size=1)
    dag.add_classical_register(size=1)
    openqasm = dag.to_openqasm()
    expected = oq_lib.openqasm_header() + oq_lib.register_initialization_string(
        [1, 1], [], [1]
    )
    check_openqasm_equivalency(openqasm, expected)


def test_gates_1(all_gate_circuit):
    all_gate_circuit.validate()
    try:
        qasm_str = all_gate_circuit.to_openqasm()
        QuantumCircuit.from_qasm_str(qasm_str)
    except Exception as e:
        print(qasm_str)
        raise e


def test_gates_2(initialization_circuit):
    initialization_circuit.validate()
    try:
        qasm_str = initialization_circuit.to_openqasm()
        QuantumCircuit.from_qasm_str(qasm_str)
    except Exception as e:
        print(qasm_str)
        raise e


@pytest.mark.parametrize(
    "gate_list",
    [
        [ops.Identity, ops.Identity],
        [ops.Identity, ops.SigmaX],
        [ops.Identity, ops.SigmaY],
        [ops.Identity, ops.SigmaZ],
        [ops.Hadamard, ops.Phase, ops.Hadamard, ops.Phase],
    ],
)
def test_gates_wrapper_1(gate_list):
    dag = CircuitDAG(n_emitter=1, n_photon=1, n_classical=1)
    op = ops.OneQubitGateWrapper(gate_list, register=0, reg_type="e")
    dag.add(op)
    try:
        qasm_str = dag.to_openqasm()
        QuantumCircuit.from_qasm_str(qasm_str)
    except Exception as e:
        print(qasm_str)
        raise e


@visualization
def test_load_circuit_1(initialization_circuit):
    """
    Tests that we can load a circuit in from our own openQASM scripts
    and retrieve the same original circuit
    """
    circuit = initialization_circuit.from_openqasm(initialization_circuit.to_openqasm())
    plot_two_circuits(initialization_circuit, circuit)


@visualization
def test_load_circuit_2():
    """
    Tests that we can load a circuit in from our own openQASM scripts
    and retrieve the same original circuit

    This time, we try to test all operations
    """
    circuit1 = CircuitDAG(n_emitter=1, n_photon=2, n_classical=2)
    circuit1.add(
        ops.OneQubitGateWrapper(
            [ops.Hadamard, ops.Phase, ops.SigmaY], register=0, reg_type="e"
        )
    )
    circuit1.add(ops.Hadamard(register=1, reg_type="p"))
    circuit1.add(ops.SigmaX(register=0, reg_type="p"))
    circuit1.add(ops.SigmaX(register=0, reg_type="e"))
    circuit1.add(ops.SigmaY(register=0, reg_type="e"))
    circuit1.add(ops.SigmaZ(register=0, reg_type="e"))
    circuit1.add(ops.Phase(register=1, reg_type="p"))
    circuit1.add(ops.Identity(register=0, reg_type="p"))
    circuit1.add(ops.CNOT(control=0, control_type="e", target=1, target_type="p"))
    circuit1.add(ops.CZ(control=0, control_type="p", target=1, target_type="p"))
    circuit1.add(
        ops.MeasurementCNOTandReset(
            control=0, control_type="e", target=0, target_type="p", c_register=1
        )
    )
    circuit1.add(ops.MeasurementZ(register=1, reg_type="p", c_register=0))
    circuit1.add(
        ops.ClassicalCNOT(
            control=0, control_type="e", target=1, target_type="p", c_register=0
        )
    )
    circuit1.add(
        ops.ClassicalCZ(
            control=0, control_type="p", target=0, target_type="e", c_register=1
        )
    )

    circuit2 = CircuitDAG.from_openqasm(circuit1.to_openqasm())
    plot_two_circuits(circuit1, circuit2)


def test_load_circuit_3(initialization_circuit):
    """
    Here, we check that we get the expected failure when a gate that we do not have defined it used
    """
    str = initialization_circuit.to_openqasm()
    str += "hshk e0[0];"
    with pytest.raises(AssertionError):
        circuit = CircuitDAG.from_openqasm(str)


def test_load_circuit_4(initialization_circuit):
    """
    Here, we check that we get the expected failure when a gate that we do not have defined it used
    """
    str = initialization_circuit.to_openqasm()
    str += "h e0[0], e1[0];"
    with pytest.raises(TypeError):
        circuit = CircuitDAG.from_openqasm(str)


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
    dag.add(ops.CNOT(control=1, control_type="e", target=0, target_type="p"))

    # Add unitary gates
    dag.add(ops.SigmaX(register=0, reg_type="e"))
    dag.add(ops.Hadamard(register=1, reg_type="p"))
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
    dag.add(ops.CNOT(control=0, control_type="e", target=1, target_type="e"))
    dag.add(ops.CZ(control=1, control_type="e", target=0, target_type="e"))
    dag.add(ops.MeasurementZ(register=0, reg_type="e", c_register=0))
    fig, ax = dag.draw_circuit(show=False)
    fig.suptitle("testing fig reception")
    plt.show()


@visualization
def test_visualization_gates_1(all_gate_circuit):
    all_gate_circuit.validate()
    try:
        all_gate_circuit.draw_circuit()
    except Exception as e:
        print(all_gate_circuit.to_openqasm())
        raise e


@visualization
def test_visualization_gates_2(initialization_circuit):
    initialization_circuit.validate()
    try:
        initialization_circuit.draw_circuit()
    except Exception as e:
        print(initialization_circuit.to_openqasm())
        raise e


@pytest.mark.parametrize(
    "gate_list",
    [
        [ops.Identity, ops.Identity],
        [ops.Identity, ops.SigmaX],
        [ops.Identity, ops.SigmaY],
        [ops.Identity, ops.SigmaZ],
        [ops.Hadamard, ops.Phase, ops.Hadamard, ops.Phase],
    ],
)
@visualization
def test_gates_wrapper_visualization_1(gate_list):
    dag = CircuitDAG(n_emitter=1, n_photon=1, n_classical=1)
    op = ops.OneQubitGateWrapper(gate_list, register=0, reg_type="e")
    dag.add(op)
    dag.draw_circuit()


@pytest.mark.parametrize(
    "gate_list",
    [
        [ops.Identity, ops.Identity],
        [ops.Identity, ops.SigmaX],
        [ops.Identity, ops.SigmaY],
        [ops.Identity, ops.SigmaZ],
        [ops.Hadamard, ops.Phase, ops.Hadamard, ops.Phase],
    ],
)
@visualization
def test_gates_wrapper_visualization_2(gate_list, initialization_circuit):
    op = ops.OneQubitGateWrapper(gate_list, register=0, reg_type="e")
    initialization_circuit.add(op)
    initialization_circuit.draw_circuit()


@pytest.mark.parametrize(
    "gate_list",
    [
        [ops.Identity, ops.Identity],
        [ops.Identity, ops.SigmaX],
        [ops.Identity, ops.SigmaY],
        [ops.Identity, ops.SigmaZ],
        [ops.Hadamard, ops.Phase, ops.Hadamard, ops.Phase],
    ],
)
@visualization
def test_gates_wrapper_visualization_3(gate_list, all_gate_circuit):
    op = ops.OneQubitGateWrapper(gate_list, register=0, reg_type="p")
    all_gate_circuit.add(op)
    all_gate_circuit.draw_circuit()


def test_parameterized_gates_1():
    # Test with the ParameterizedOneQubitRotation and ParameterizedControlledRotationQubit
    dag = CircuitDAG(n_emitter=1, n_photon=1, n_classical=1)
    param_info = {"theta": np.pi, "phi": np.pi / 2, "lambda": np.pi / 4}
    dag.add(
        ops.ParameterizedOneQubitRotation(
            register=0, reg_type="p", param_info=param_info
        )
    )
    dag.add(
        ops.ParameterizedControlledRotationQubit(
            control=0,
            control_type="p",
            target=1,
            target_type="p",
            param_info=param_info,
        )
    )
    dag.validate()
    dag.draw_circuit()


def test_parameterized_gates_2():
    # Use the RY, RZ, RX ates
    dag = CircuitDAG(n_emitter=1, n_photon=1, n_classical=1)
    param_info = {"theta": np.pi, "phi": np.pi / 2, "lambda": np.pi / 4}
    dag.add(ops.RY(register=0, reg_type="p", param_info=param_info))
    dag.add(ops.RZ(register=0, reg_type="p", param_info=param_info))
    dag.add(ops.RX(register=0, reg_type="p", param_info=param_info))
    dag.validate()
    dag.draw_circuit()


def test_parameterized_gates_3():
    # Using many gates and adding parameterized gates to the circuit
    dag = CircuitDAG(n_emitter=1, n_photon=3, n_classical=1)
    # param_info = {"theta": np.pi}
    param_info = {"theta": np.pi, "phi": np.pi / 2, "lambda": np.pi / 4}
    dag.add(
        ops.ParameterizedOneQubitRotation(
            register=0, reg_type="e", param_info=param_info
        )
    )
    dag.add(
        ops.ParameterizedOneQubitRotation(
            register=0, reg_type="e", param_info=param_info
        )
    )
    dag.add(ops.CNOT(control=1, control_type="e", target=0, target_type="e"))
    dag.add(
        ops.ParameterizedOneQubitRotation(
            register=0, reg_type="p", param_info=param_info
        )
    )

    dag.add(
        ops.ParameterizedControlledRotationQubit(
            control=0,
            control_type="e",
            target=1,
            target_type="e",
            param_info=param_info,
        )
    )
    dag.add(
        ops.ParameterizedControlledRotationQubit(
            control=0,
            control_type="e",
            target=1,
            target_type="e",
            param_info=param_info,
        )
    )
    # add hadmard gate
    dag.add(ops.Hadamard(register=0, reg_type="e"))
    dag.add(ops.Hadamard(register=0, reg_type="p"))

    # add measurement
    dag.add(ops.MeasurementZ(register=0, reg_type="e", c_register=0))
    dag.add(ops.MeasurementZ(register=0, reg_type="p", c_register=0))
    dag.add(ops.MeasurementZ(register=1, reg_type="e", c_register=0))
    # param_info = {"theta": np.pi}
    dag.add(ops.RY(register=0, reg_type="p", param_info=param_info))
    dag.add(ops.RX(register=1, reg_type="p", param_info=param_info))
    dag.add(ops.RZ(register=0, reg_type="p", param_info=param_info))

    dag.validate()
    dag.draw_circuit()


def test_parameterized_gates_4():
    # Using many gates and adding parameterized gates to the circuit

    circuit1 = CircuitDAG(n_emitter=1, n_photon=2, n_classical=2)
    circuit1.add(
        ops.OneQubitGateWrapper(
            [ops.Hadamard, ops.Phase, ops.SigmaY], register=0, reg_type="e"
        )
    )
    circuit1.add(ops.Hadamard(register=1, reg_type="p"))
    circuit1.add(ops.SigmaX(register=0, reg_type="p"))
    circuit1.add(ops.SigmaX(register=0, reg_type="e"))
    circuit1.add(ops.SigmaY(register=0, reg_type="e"))
    circuit1.add(ops.SigmaZ(register=0, reg_type="e"))
    circuit1.add(ops.Phase(register=1, reg_type="p"))
    circuit1.add(ops.Identity(register=0, reg_type="p"))
    circuit1.add(ops.CNOT(control=0, control_type="e", target=1, target_type="p"))
    circuit1.add(ops.CZ(control=0, control_type="p", target=1, target_type="p"))
    circuit1.add(
        ops.MeasurementCNOTandReset(
            control=0, control_type="e", target=0, target_type="p", c_register=1
        )
    )
    circuit1.add(ops.MeasurementZ(register=1, reg_type="p", c_register=0))
    circuit1.add(
        ops.ClassicalCNOT(
            control=0, control_type="e", target=1, target_type="p", c_register=0
        )
    )
    circuit1.add(
        ops.ClassicalCZ(
            control=0, control_type="p", target=0, target_type="e", c_register=1
        )
    )
    param_info = {"theta": np.pi, "phi": np.pi / 2, "lambda": np.pi / 4}
    circuit1.add(
        ops.ParameterizedOneQubitRotation(
            register=0, reg_type="p", param_info=param_info
        )
    )
    circuit1.add(
        ops.ParameterizedOneQubitRotation(
            register=1, reg_type="p", param_info=param_info
        )
    )
    circuit1.add(ops.RY(register=0, reg_type="p", param_info=param_info))
    circuit1.add(ops.RX(register=1, reg_type="p", param_info=param_info))
    circuit1.add(ops.RZ(register=0, reg_type="p", param_info=param_info))
    # increase the  text size of the angle labels
    circuit1.draw_circuit()
