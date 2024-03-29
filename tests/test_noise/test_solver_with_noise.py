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
import matplotlib.pyplot as plt
import pytest
import numpy as np

from graphiq.benchmarks.circuits import (
    linear_cluster_3qubit_circuit,
    linear_cluster_4qubit_circuit,
    ghz3_state_circuit,
)
import graphiq.noise.noise_models as nm
from graphiq.backends.density_matrix.compiler import DensityMatrixCompiler
import graphiq.backends.density_matrix.functions as dmf
from graphiq.metrics import Infidelity, TraceDistance
from graphiq.solvers.evolutionary_solver import EvolutionarySolver
from graphiq.visualizers.density_matrix import density_matrix_bars
from tests.test_flags import visualization


@pytest.fixture(scope="module")
def density_matrix_compiler():
    """
    Here, we set the scope to module because, while we want to use a DensityMatrixCompiler multiple times, we don't
    actually need a different copy each time
    """
    compiler = DensityMatrixCompiler()
    compiler.measurement_determinism = 1
    return compiler


def generate_run_noise(
    n_photon, n_emitter, expected_triple, compiler, noise_model_mapping, seed
):
    target, _, metric = expected_triple

    solver = EvolutionarySolver(
        target=target,
        metric=metric,
        compiler=compiler,
        n_emitter=n_emitter,
        n_photon=n_photon,
        noise_model_mapping=noise_model_mapping,
    )
    solver.seed(seed)
    solver.solve()

    state = compiler.compile(solver.hof[0][1])

    state.partial_trace(list(range(n_photon)), (n_photon + n_emitter) * [2])
    return solver.hof, state


def generate_run_no_noise(n_photon, n_emitter, expected_triple, compiler, seed):
    target, _, metric = expected_triple

    solver = EvolutionarySolver(
        target=target,
        metric=metric,
        compiler=compiler,
        n_emitter=n_emitter,
        n_photon=n_photon,
    )
    solver.seed(seed)
    solver.solve()

    state = compiler.compile(solver.hof[0][1])

    state.partial_trace(list(range(n_photon)), (n_photon + n_emitter) * [2])
    return solver.hof, state


def check_run(run_info, expected_info):
    hof, state = run_info
    target_state, _, metric = expected_info

    circuit = hof[0][1]
    assert np.isclose(hof[0][0], metric.evaluate(state, circuit))


def check_run_visual(run_info, expected_info):
    hof, state = run_info
    target_state, _, metric = expected_info

    circuit = hof[0][1]
    circuit.draw_circuit()
    fig, axs = density_matrix_bars(target_state.rep_data.data)
    fig.suptitle("TARGET DENSITY MATRIX")
    plt.show()

    fig, axs = density_matrix_bars(state.rep_data.data)
    fig.suptitle("CREATED DENSITY MATRIX")
    plt.show()


def check_run_noise_visual(no_noise_run_info, noise_run_info, expected_info):
    hof_no_noise, state_no_noise = no_noise_run_info
    hof_noise, state_noise = noise_run_info
    target_state, _, metric = expected_info

    circuit_no_noise = hof_no_noise[0][1]
    circuit_no_noise.draw_circuit()

    fig, axs = density_matrix_bars(target_state.rep_data.data)
    fig.suptitle("TARGET DENSITY MATRIX")
    plt.show()

    fig, axs = density_matrix_bars(state_no_noise.rep_data.data)
    fig.suptitle("CREATED DENSITY MATRIX W/O NOISE")
    plt.show()

    circuit_noise = hof_noise[0][1]
    circuit_noise.draw_circuit()

    fig, axs = density_matrix_bars(state_noise.rep_data.data)
    fig.suptitle("CREATED DENSITY MATRIX WITH NOISE")
    plt.show()


@pytest.fixture(scope="module")
def linear3_run_noise(density_matrix_compiler, linear3_expected, linear3_noise_model_2):
    """
    Again, we set the fixture scope to module. Arguably, this is more important than last time because actually
    running the solve function takes (relatively) long.

    Since we want to apply 2 separate tests on the same run (one visual, one non-visual), it makes sense to have a
    common fixture that only gets called once per module
    """
    return generate_run_noise(
        3, 1, linear3_expected, density_matrix_compiler, linear3_noise_model_2, 10
    )


@pytest.fixture(scope="module")
def linear3_run_no_noise(density_matrix_compiler, linear3_expected):
    """
    Again, we set the fixture scope to module. Arguably, this is more important than last time because actually
    running the solve takes (relatively) long.

    Since we want to apply 2 separate tests on the same run (one visual, one non-visual), it makes sense to have a
    common fixture that only gets called once per module
    """
    return generate_run_no_noise(3, 1, linear3_expected, density_matrix_compiler, 10)


@pytest.fixture(scope="module")
def linear3_run_trace_distance(
    density_matrix_compiler,
    linear3_expected_trace_distance,
    linear3_noise_model,
):
    """
    Again, we set the fixture scope to module. Arguably, this is more important than last time because actually
    running the solve takes (relatively) long.

    Since we want to apply 2 separate tests on the same run (one visual, one non-visual), it makes sense to have a
    common fixture that only gets called once per module
    """
    return generate_run_noise(
        3,
        1,
        linear3_expected_trace_distance,
        density_matrix_compiler,
        linear3_noise_model,
        10,
    )


@pytest.fixture(scope="module")
def linear3_noise_model():
    noise_model_mapping = dict()
    noise_model_mapping["e"] = {
        "Identity": nm.OneQubitGateReplacement(
            dmf.parameterized_one_qubit_unitary(np.pi / 180, 0, 0)
        ),
        "SigmaX": nm.DepolarizingNoise(0.01),
    }
    noise_model_mapping["ee"] = {
        "Identity": nm.OneQubitGateReplacement(
            dmf.parameterized_one_qubit_unitary(np.pi / 180, 0, 0)
        ),
        "SigmaX": nm.DepolarizingNoise(0.01),
        "CNOT": nm.DepolarizingNoise(0.01),
    }
    noise_model_mapping["ep"] = {"CNOT": nm.DepolarizingNoise(0.01)}
    noise_model_mapping["p"] = {}
    return noise_model_mapping


@pytest.fixture(scope="module")
def linear3_expected():
    circuit_ideal, target_state = linear_cluster_3qubit_circuit()

    metric = Infidelity(target=target_state)

    return target_state, circuit_ideal, metric


@pytest.fixture(scope="module")
def linear3_expected_trace_distance():
    circuit_ideal, target_state = linear_cluster_3qubit_circuit()

    metric = TraceDistance(target=target_state)
    return target_state, circuit_ideal, metric


@pytest.fixture(scope="module")
def linear4_run_noise(density_matrix_compiler, linear4_expected, linear4_noise_model):
    """
    Again, we set the fixture scope to module. Arguably, this is more important than last time because actually
    running the solve function takes (relatively) long.

    Since we want to apply 2 separate tests on the same run (one visual, one non-visual), it makes sense to have a
    common fixture that only gets called once per module
    """
    return generate_run_noise(
        4, 1, linear4_expected, density_matrix_compiler, linear4_noise_model, 10
    )


@pytest.fixture(scope="module")
def linear4_run_no_noise(density_matrix_compiler, linear4_expected):
    """
    Again, we set the fixture scope to module. Arguably, this is more important than last time because actually
    running the solve function takes (relatively) long.

    Since we want to apply 2 separate tests on the same run (one visual, one non-visual), it makes sense to have a
    common fixture that only gets called once per module
    """
    return generate_run_no_noise(4, 1, linear4_expected, density_matrix_compiler, 10)


@pytest.fixture(scope="module")
def linear4_run_trace_distance(
    density_matrix_compiler,
    linear4_expected_trace_distance,
    linear4_noise_model,
):
    """
    Again, we set the fixture scope to module. Arguably, this is more important than last time because actually
    running the solve function takes (relatively) long.

    Since we want to apply 2 separate tests on the same run (one visual, one non-visual), it makes sense to have a
    common fixture that only gets called once per module
    """
    return generate_run_noise(
        4,
        1,
        linear4_expected_trace_distance,
        density_matrix_compiler,
        linear4_noise_model,
        10,
    )


@pytest.fixture(scope="module")
def linear4_expected():
    circuit_ideal, target_state = linear_cluster_4qubit_circuit()

    metric = Infidelity(target=target_state)

    return target_state, circuit_ideal, metric


@pytest.fixture(scope="module")
def linear4_expected_trace_distance():
    circuit_ideal, target_state = linear_cluster_4qubit_circuit()

    metric = TraceDistance(target=target_state)

    return target_state, circuit_ideal, metric


@pytest.fixture(scope="module")
def ghz3_run_noise(density_matrix_compiler, ghz3_expected, ghz3_noise_model):
    return generate_run_noise(
        3, 1, ghz3_expected, density_matrix_compiler, ghz3_noise_model, 0
    )


@pytest.fixture(scope="module")
def ghz3_run_no_noise(density_matrix_compiler, ghz3_expected):
    return generate_run_no_noise(3, 1, ghz3_expected, density_matrix_compiler, 0)


@pytest.fixture(scope="module")
def ghz3_expected():
    circuit_ideal, target_state = ghz3_state_circuit()

    metric = Infidelity(target=target_state)

    return target_state, circuit_ideal, metric


def test_solver_linear3(linear3_run_noise, linear3_expected):
    check_run(linear3_run_noise, linear3_expected)


def test_solver_linear3_trace_distance(
    linear3_run_trace_distance, linear3_expected_trace_distance
):
    check_run(linear3_run_trace_distance, linear3_expected_trace_distance)


@pytest.fixture(scope="module")
def linear3_noise_model_2():
    dp = 0.03
    noise_model_mapping = dict()
    noise_model_mapping["e"] = {
        "Identity": nm.DepolarizingNoise(dp),
        "Hadamard": nm.DepolarizingNoise(dp),
    }
    noise_model_mapping["ee"] = {"CNOT": nm.DepolarizingNoise(dp)}
    noise_model_mapping["ep"] = {
        "CNOT": nm.DepolarizingNoise(dp),
        "MeasurementCNOTandRest": nm.DepolarizingNoise(dp),
    }
    noise_model_mapping["p"] = {
        "Identity": nm.DepolarizingNoise(dp),
        "Hadamard": nm.DepolarizingNoise(dp),
    }

    return noise_model_mapping


@visualization
def test_solver_linear3_visualized(
    linear3_run_no_noise, linear3_run_noise, linear3_expected
):
    check_run_noise_visual(linear3_run_no_noise, linear3_run_noise, linear3_expected)


@pytest.fixture(scope="module")
def linear4_noise_model():
    dp = 0.1
    noise_model_mapping = dict()
    noise_model_mapping["e"] = {
        "Identity": nm.DepolarizingNoise(dp),
        "Hadamard": nm.DepolarizingNoise(dp),
        "Phase": nm.DepolarizingNoise(0.1),
    }
    noise_model_mapping["ee"] = {"CNOT": nm.DepolarizingNoise(dp)}
    noise_model_mapping["ep"] = {
        "CNOT": nm.DepolarizingNoise(dp),
        "MeasurementCNOTandRest": nm.DepolarizingNoise(dp),
    }
    noise_model_mapping["p"] = {
        "Identity": nm.DepolarizingNoise(dp),
        "Hadamard": nm.DepolarizingNoise(dp),
        "Phase": nm.DepolarizingNoise(0.1),
    }
    return noise_model_mapping


def test_solver_linear4(linear4_run_noise, linear4_expected):
    check_run(linear4_run_noise, linear4_expected)


def test_solver_linear4_trace_distance(
    linear4_run_trace_distance, linear4_expected_trace_distance
):
    check_run(linear4_run_trace_distance, linear4_expected_trace_distance)


@visualization
def test_solver_linear4_visualized(
    linear4_run_no_noise, linear4_run_noise, linear4_expected
):
    check_run_noise_visual(linear4_run_no_noise, linear4_run_noise, linear4_expected)


@pytest.fixture(scope="module")
def ghz3_noise_model():
    noise_model_mapping = dict()
    noise_model_mapping["e"] = {
        "Identity": nm.PauliError("X"),
        "SigmaX": nm.DepolarizingNoise(0.01),
        "Phase": nm.PhasePerturbedError(0, 0, np.pi / 180),
    }
    noise_model_mapping["ee"] = {"CNOT": nm.DepolarizingNoise(0.01)}
    noise_model_mapping["ep"] = {"CNOT": nm.DepolarizingNoise(0.01)}
    noise_model_mapping["p"] = {
        "Identity": nm.PauliError("X"),
        "SigmaX": nm.DepolarizingNoise(0.01),
        "Phase": nm.PhasePerturbedError(0, 0, np.pi / 180),
    }

    return noise_model_mapping


@visualization
def test_solver_ghz3_visualized(ghz3_run_no_noise, ghz3_run_noise, ghz3_expected):
    check_run_noise_visual(ghz3_run_no_noise, ghz3_run_noise, ghz3_expected)
