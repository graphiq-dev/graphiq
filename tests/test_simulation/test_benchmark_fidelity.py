import pytest
import numpy as np

import benchmarks.circuits as circ
import benchmarks.circuits as equiv_circ
from src.metrics import Infidelity
from src.backends.density_matrix.compiler import DensityMatrixCompiler

from benchmarks.benchmark_utils import circuit_measurement_independent

@pytest.mark.parametrize("ghz3_state_circuit", [circ.ghz3_state_circuit, equiv_circ.ghz3_state_circuit])
def test_ghz3(ghz3_state_circuit):
    target_circuit, ideal_state = ghz3_state_circuit()
    compiler = DensityMatrixCompiler()
    independent, state1, state2 = circuit_measurement_independent(target_circuit, compiler)
    assert independent

    infidelity = Infidelity(ideal_state['dm'])
    print(f'state1: {state1}')
    print(f'state2: {state2}')
    assert np.isclose(infidelity.evaluate(state1, target_circuit), 0.0)
    assert np.isclose(infidelity.evaluate(state2, target_circuit), 0.0)


@pytest.mark.parametrize("ghz4_state_circuit", [circ.ghz4_state_circuit,
                                                equiv_circ.ghz4_state_circuit])
def test_ghz4(ghz4_state_circuit):
    target_circuit, ideal_state = ghz4_state_circuit()
    compiler = DensityMatrixCompiler()
    independent, state1, state2 = circuit_measurement_independent(target_circuit, compiler)
    assert independent

    infidelity = Infidelity(ideal_state['dm'])
    assert np.isclose(infidelity.evaluate(state1, target_circuit), 0.0)
    assert np.isclose(infidelity.evaluate(state2, target_circuit), 0.0)


@pytest.mark.parametrize("linear_cluster_3qubit_circuit", [circ.linear_cluster_3qubit_circuit,
                                                           equiv_circ.linear_cluster_3qubit_circuit])
def test_linear_3(linear_cluster_3qubit_circuit):
    target_circuit, ideal_state = linear_cluster_3qubit_circuit()
    compiler = DensityMatrixCompiler()
    independent, state1, state2 = circuit_measurement_independent(target_circuit, compiler)
    assert independent

    infidelity = Infidelity(ideal_state['dm'])
    assert np.isclose(infidelity.evaluate(state1, target_circuit), 0.0)
    assert np.isclose(infidelity.evaluate(state2, target_circuit), 0.0)


@pytest.mark.parametrize("linear_cluster_4qubit_circuit", [circ.linear_cluster_4qubit_circuit,
                                                           equiv_circ.linear_cluster_4qubit_circuit])
def test_linear_4(linear_cluster_4qubit_circuit):
    target_circuit, ideal_state = linear_cluster_4qubit_circuit()
    compiler = DensityMatrixCompiler()
    independent, state1, state2 = circuit_measurement_independent(target_circuit, compiler)
    assert independent

    infidelity = Infidelity(ideal_state['dm'])
    assert np.isclose(infidelity.evaluate(state1, target_circuit), 0.0)
    assert np.isclose(infidelity.evaluate(state2, target_circuit), 0.0)
