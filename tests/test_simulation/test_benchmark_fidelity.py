import pytest
import numpy as np

import benchmarks.circuits_original as og_circ
import benchmarks.circuits as equiv_circ
from src.metrics import MetricFidelity
from src.backends.density_matrix.compiler import DensityMatrixCompiler
from src.backends.density_matrix.functions import partial_trace


def circuit_measurement_independent(circuit, compiler):
    # Note: this doesn't check every option, only "all 0s, all 1s"

    state1 = compiler.compile(circuit, set_measurement=0)
    state2 = compiler.compile(circuit, set_measurement=1)

    # circuit.draw_circuit()
    # state1.draw()
    # state2.draw()

    state_data1 = partial_trace(state1.data, keep=list(range(0, circuit.n_quantum - 1)), dims=circuit.n_quantum * [2])
    state_data2 = partial_trace(state2.data, keep=list(range(0, circuit.n_quantum - 1)), dims=circuit.n_quantum * [2])

    assert np.allclose(state_data1, state_data2), f'state1: {state_data1}, \n state2: {state_data2}'
    return state_data1, state_data2


@pytest.mark.parametrize("ghz3_state_circuit", [og_circ.ghz3_state_circuit, equiv_circ.ghz3_state_circuit])
def test_ghz3(ghz3_state_circuit):
    target_circuit, ideal_state = ghz3_state_circuit()
    compiler = DensityMatrixCompiler()
    state1, state2 = circuit_measurement_independent(target_circuit, compiler)

    infidelity = MetricFidelity(ideal_state['dm'])
    assert np.isclose(infidelity.evaluate(state1, target_circuit), 0.0)
    assert np.isclose(infidelity.evaluate(state2, target_circuit), 0.0)


@pytest.mark.parametrize("ghz4_state_circuit", [og_circ.ghz4_state_circuit, equiv_circ.ghz4_state_circuit])
def test_ghz4(ghz4_state_circuit):
    target_circuit, ideal_state = ghz4_state_circuit()
    compiler = DensityMatrixCompiler()
    state1, state2 = circuit_measurement_independent(target_circuit, compiler)

    infidelity = MetricFidelity(ideal_state['dm'])
    assert np.isclose(infidelity.evaluate(state1, target_circuit), 0.0)
    assert np.isclose(infidelity.evaluate(state2, target_circuit), 0.0)


@pytest.mark.parametrize("linear_cluster_3qubit_circuit", [og_circ.linear_cluster_3qubit_circuit, equiv_circ.linear_cluster_3qubit_circuit])
def test_linear_3(linear_cluster_3qubit_circuit):
    target_circuit, ideal_state = linear_cluster_3qubit_circuit()
    compiler = DensityMatrixCompiler()
    state1, state2 = circuit_measurement_independent(target_circuit, compiler)

    infidelity = MetricFidelity(ideal_state['dm'])
    assert np.isclose(infidelity.evaluate(state1, target_circuit), 0.0)
    assert np.isclose(infidelity.evaluate(state2, target_circuit), 0.0)


@pytest.mark.parametrize("linear_cluster_4qubit_circuit", [og_circ.linear_cluster_4qubit_circuit, equiv_circ.linear_cluster_4qubit_circuit])
def test_linear_4(linear_cluster_4qubit_circuit):
    target_circuit, ideal_state = linear_cluster_4qubit_circuit()
    compiler = DensityMatrixCompiler()
    state1, state2 = circuit_measurement_independent(target_circuit, compiler)

    infidelity = MetricFidelity(ideal_state['dm'])
    assert np.isclose(infidelity.evaluate(state1, target_circuit), 0.0)
    assert np.isclose(infidelity.evaluate(state2, target_circuit), 0.0)
