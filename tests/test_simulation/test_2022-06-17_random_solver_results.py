import pytest
import numpy as np

from src.circuit import CircuitDAG
import src.ops as ops
from src.backends.density_matrix.compiler import DensityMatrixCompiler
from src.metrics import Infidelity
from benchmarks.benchmark_utils import circuit_measurement_independent
from benchmarks.circuits import (
    ghz3_state_circuit,
    ghz4_state_circuit,
    linear_cluster_3qubit_circuit,
    linear_cluster_4qubit_circuit,
)


def ghz3_0():
    dag = CircuitDAG(n_emitter=1, n_photon=3, n_classical=1)
    dag.add(ops.Hadamard(register=0, reg_type="e"))
    dag.add(ops.CNOT(control=0, control_type="e", target=0, target_type="p"))
    dag.add(ops.CNOT(control=0, control_type="e", target=1, target_type="p"))
    dag.add(ops.Hadamard(register=0, reg_type="e"))
    dag.add(ops.CNOT(control=0, control_type="e", target=2, target_type="p"))
    dag.add(ops.Hadamard(register=0, reg_type="e"))
    dag.add(ops.Hadamard(register=2, reg_type="p"))
    dag.add(
        ops.MeasurementCNOTandReset(
            control=0, control_type="e", target=2, target_type="p", c_register=0
        )
    )
    return dag


def ghz3_1():
    dag = CircuitDAG(n_emitter=1, n_photon=3, n_classical=1)
    dag.add(ops.Hadamard(register=0, reg_type="e"))
    dag.add(ops.CNOT(control=0, control_type="e", target=0, target_type="p"))
    dag.add(ops.CNOT(control=0, control_type="e", target=1, target_type="p"))
    dag.add(ops.Hadamard(register=0, reg_type="e"))
    dag.add(ops.CNOT(control=0, control_type="e", target=2, target_type="p"))
    dag.add(ops.Hadamard(register=2, reg_type="p"))
    dag.add(
        ops.SingleQubitGateWrapper(
            [ops.Hadamard, ops.Phase, ops.SigmaZ], register=0, reg_type="e"
        )
    )
    dag.add(
        ops.MeasurementCNOTandReset(
            control=0, control_type="e", target=2, target_type="p", c_register=0
        )
    )
    return dag


def ghz3_2():
    dag = CircuitDAG(n_emitter=1, n_photon=3, n_classical=1)
    dag.add(ops.Hadamard(register=0, reg_type="e"))
    dag.add(ops.CNOT(control=0, control_type="e", target=0, target_type="p"))
    dag.add(ops.Hadamard(register=0, reg_type="e"))
    dag.add(ops.Hadamard(register=0, reg_type="p"))
    dag.add(ops.CNOT(control=0, control_type="e", target=1, target_type="p"))
    dag.add(ops.Hadamard(register=0, reg_type="e"))
    dag.add(ops.CNOT(control=0, control_type="e", target=2, target_type="p"))
    dag.add(ops.Hadamard(register=0, reg_type="e"))
    dag.add(ops.Hadamard(register=2, reg_type="p"))
    dag.add(
        ops.MeasurementCNOTandReset(
            control=0, control_type="e", target=2, target_type="p", c_register=0
        )
    )
    return dag


def ghz4_0():
    dag = CircuitDAG(n_emitter=1, n_photon=4, n_classical=1)
    dag.add(ops.Hadamard(register=0, reg_type="e"))
    dag.add(ops.CNOT(control=0, control_type="e", target=0, target_type="p"))
    dag.add(ops.CNOT(control=0, control_type="e", target=1, target_type="p"))
    dag.add(ops.CNOT(control=0, control_type="e", target=2, target_type="p"))
    dag.add(ops.Hadamard(register=0, reg_type="e"))
    dag.add(ops.CNOT(control=0, control_type="e", target=3, target_type="p"))
    dag.add(ops.Hadamard(register=0, reg_type="e"))
    dag.add(ops.Hadamard(register=3, reg_type="p"))
    dag.add(
        ops.MeasurementCNOTandReset(
            control=0, control_type="e", target=3, target_type="p", c_register=0
        )
    )
    return dag


def ghz4_1():
    dag = CircuitDAG(n_emitter=1, n_photon=4, n_classical=1)
    dag.add(ops.Hadamard(register=0, reg_type="e"))
    dag.add(ops.CNOT(control=0, control_type="e", target=0, target_type="p"))
    dag.add(ops.Hadamard(register=0, reg_type="e"))
    dag.add(ops.Hadamard(register=0, reg_type="p"))
    dag.add(ops.CNOT(control=0, control_type="e", target=1, target_type="p"))
    dag.add(ops.CNOT(control=0, control_type="e", target=2, target_type="p"))
    dag.add(ops.Hadamard(register=0, reg_type="e"))
    dag.add(ops.CNOT(control=0, control_type="e", target=3, target_type="p"))
    dag.add(ops.Hadamard(register=0, reg_type="e"))
    dag.add(ops.Hadamard(register=3, reg_type="p"))
    dag.add(
        ops.MeasurementCNOTandReset(
            control=0, control_type="e", target=3, target_type="p", c_register=0
        )
    )
    return dag


def linear3_0():
    dag = CircuitDAG(n_emitter=1, n_photon=3, n_classical=1)
    dag.add(ops.Hadamard(register=0, reg_type="e"))
    dag.add(ops.CNOT(control=0, control_type="e", target=0, target_type="p"))
    dag.add(ops.Hadamard(register=0, reg_type="p"))
    dag.add(ops.CNOT(control=0, control_type="e", target=1, target_type="p"))
    dag.add(ops.CNOT(control=0, control_type="e", target=2, target_type="p"))
    dag.add(ops.Hadamard(register=0, reg_type="e"))
    dag.add(ops.Hadamard(register=2, reg_type="p"))
    dag.add(
        ops.MeasurementCNOTandReset(
            control=0, control_type="e", target=0, target_type="p", c_register=0
        )
    )
    return dag


def linear3_1():
    dag = CircuitDAG(n_emitter=1, n_photon=3, n_classical=1)
    dag.add(ops.Hadamard(register=0, reg_type="e"))
    dag.add(ops.CNOT(control=0, control_type="e", target=0, target_type="p"))
    dag.add(ops.Hadamard(register=0, reg_type="p"))
    dag.add(ops.CNOT(control=0, control_type="e", target=1, target_type="p"))
    dag.add(ops.CNOT(control=0, control_type="e", target=2, target_type="p"))
    dag.add(ops.Hadamard(register=0, reg_type="e"))
    dag.add(ops.Hadamard(register=2, reg_type="p"))
    dag.add(
        ops.MeasurementCNOTandReset(
            control=0, control_type="e", target=2, target_type="p", c_register=0
        )
    )
    return dag


def linear4_0():
    dag = CircuitDAG(n_emitter=1, n_photon=4, n_classical=1)
    dag.add(ops.Hadamard(register=0, reg_type="e"))
    dag.add(ops.CNOT(control=0, control_type="e", target=0, target_type="p"))
    dag.add(ops.Hadamard(register=0, reg_type="e"))
    dag.add(ops.CNOT(control=0, control_type="e", target=1, target_type="p"))
    dag.add(ops.Hadamard(register=0, reg_type="e"))
    dag.add(ops.CNOT(control=0, control_type="e", target=2, target_type="p"))
    dag.add(ops.CNOT(control=0, control_type="e", target=3, target_type="p"))
    dag.add(ops.Hadamard(register=0, reg_type="e"))
    dag.add(ops.Hadamard(register=3, reg_type="p"))
    dag.add(
        ops.MeasurementCNOTandReset(
            control=0, control_type="e", target=3, target_type="p", c_register=0
        )
    )
    return dag


def linear4_1():
    dag = CircuitDAG(n_emitter=1, n_photon=4, n_classical=1)
    dag.add(ops.Hadamard(register=0, reg_type="e"))
    dag.add(ops.CNOT(control=0, control_type="e", target=0, target_type="p"))
    dag.add(ops.Hadamard(register=0, reg_type="p"))
    dag.add(ops.CNOT(control=0, control_type="e", target=1, target_type="p"))
    dag.add(ops.Hadamard(register=0, reg_type="e"))
    dag.add(ops.CNOT(control=0, control_type="e", target=2, target_type="p"))
    dag.add(ops.CNOT(control=0, control_type="e", target=3, target_type="p"))
    dag.add(ops.Hadamard(register=0, reg_type="e"))
    dag.add(ops.Hadamard(register=3, reg_type="p"))
    dag.add(
        ops.MeasurementCNOTandReset(
            control=0, control_type="e", target=3, target_type="p", c_register=0
        )
    )
    return dag


@pytest.mark.parametrize("ghz3", [ghz3_0(), ghz3_1(), ghz3_2()])
def test_ghz3(ghz3):
    demo_circuit, ideal_state = ghz3_state_circuit()
    compiler = DensityMatrixCompiler()
    independent, state1, state2 = circuit_measurement_independent(ghz3, compiler)
    assert independent

    infidelity = Infidelity(ideal_state["dm"])
    print(f"state1: {state1}")
    print(f"state2: {state2}")
    assert np.isclose(infidelity.evaluate(state1, ghz3), 0.0), f"state 1: {state1}"
    assert np.isclose(infidelity.evaluate(state2, ghz3), 0.0), f"state 2: {state1}"


@pytest.mark.parametrize("ghz4", [ghz4_0(), ghz4_1()])
def test_ghz4(ghz4):
    demo_circuit, ideal_state = ghz4_state_circuit()
    compiler = DensityMatrixCompiler()
    independent, state1, state2 = circuit_measurement_independent(ghz4, compiler)
    assert independent

    infidelity = Infidelity(ideal_state["dm"])
    assert np.isclose(infidelity.evaluate(state1, ghz4), 0.0), f"state 1: {state1}"
    assert np.isclose(infidelity.evaluate(state2, ghz4), 0.0), f"state 2: {state1}"


@pytest.mark.parametrize("linear3", [linear3_0(), linear3_1()])
def test_linear3(linear3):
    demo_circuit, ideal_state = linear_cluster_3qubit_circuit()
    compiler = DensityMatrixCompiler()
    independent, state1, state2 = circuit_measurement_independent(linear3, compiler)
    assert independent

    infidelity = Infidelity(ideal_state["dm"])
    assert np.isclose(infidelity.evaluate(state1, linear3), 0.0), f"state 1: {state1}"
    assert np.isclose(infidelity.evaluate(state2, linear3), 0.0), f"state 2: {state1}"


@pytest.mark.parametrize("linear4", [linear4_0(), linear4_1()])
def test_linear4(linear4):
    demo_circuit, ideal_state = linear_cluster_4qubit_circuit()
    compiler = DensityMatrixCompiler()
    independent, state1, state2 = circuit_measurement_independent(linear4, compiler)
    assert independent

    infidelity = Infidelity(ideal_state["dm"])
    assert np.isclose(infidelity.evaluate(state1, linear4), 0.0), f"state 1: {state1}"
    assert np.isclose(infidelity.evaluate(state2, linear4), 0.0), f"state 2: {state1}"
