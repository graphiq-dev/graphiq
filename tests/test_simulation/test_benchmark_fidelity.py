# Copyright (c) 2022-2024 Quantum Bridge Technologies Inc.
# Copyright (c) 2022-2024 Ki3 Photonics Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import numpy as np
import pytest

from graphiq.benchmarks import circuits as circ
from graphiq.benchmarks.benchmark_utils import circuit_measurement_independent
from graphiq.backends.density_matrix.compiler import DensityMatrixCompiler
from graphiq.metrics import Infidelity


@pytest.mark.parametrize("ghz3_state_circuit", [circ.ghz3_state_circuit])
def test_ghz3(ghz3_state_circuit):
    target_circuit, ideal_state = ghz3_state_circuit()

    compiler = DensityMatrixCompiler()
    independent, state1, state2 = circuit_measurement_independent(
        target_circuit, compiler
    )
    assert independent

    infidelity = Infidelity(ideal_state)
    print(f"state1: {state1}")
    print(f"state2: {state2}")
    assert np.isclose(infidelity.evaluate(state1, target_circuit), 0.0)
    assert np.isclose(infidelity.evaluate(state2, target_circuit), 0.0)


@pytest.mark.parametrize("ghz4_state_circuit", [circ.ghz4_state_circuit])
def test_ghz4(ghz4_state_circuit):
    target_circuit, ideal_state = ghz4_state_circuit()

    compiler = DensityMatrixCompiler()
    independent, state1, state2 = circuit_measurement_independent(
        target_circuit, compiler
    )
    assert independent

    infidelity = Infidelity(ideal_state)
    assert np.isclose(infidelity.evaluate(state1, target_circuit), 0.0)
    assert np.isclose(infidelity.evaluate(state2, target_circuit), 0.0)


@pytest.mark.parametrize(
    "linear_cluster_3qubit_circuit",
    [circ.linear_cluster_3qubit_circuit],
)
def test_linear_3(linear_cluster_3qubit_circuit):
    target_circuit, ideal_state = linear_cluster_3qubit_circuit()

    compiler = DensityMatrixCompiler()
    independent, state1, state2 = circuit_measurement_independent(
        target_circuit, compiler
    )
    assert independent

    infidelity = Infidelity(ideal_state)
    assert np.isclose(infidelity.evaluate(state1, target_circuit), 0.0)
    assert np.isclose(infidelity.evaluate(state2, target_circuit), 0.0)


@pytest.mark.parametrize(
    "linear_cluster_4qubit_circuit",
    [circ.linear_cluster_4qubit_circuit],
)
def test_linear_4(linear_cluster_4qubit_circuit):
    target_circuit, ideal_state = linear_cluster_4qubit_circuit()

    compiler = DensityMatrixCompiler()
    independent, state1, state2 = circuit_measurement_independent(
        target_circuit, compiler
    )
    assert independent

    infidelity = Infidelity(ideal_state)
    assert np.isclose(infidelity.evaluate(state1, target_circuit), 0.0)
    assert np.isclose(infidelity.evaluate(state2, target_circuit), 0.0)
