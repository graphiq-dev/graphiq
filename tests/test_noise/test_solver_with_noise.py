import pytest
from tests.test_flags import visualization

from src.solvers.evolutionary_solver import EvolutionarySolver
import matplotlib.pyplot as plt
from src.backends.density_matrix.compiler import DensityMatrixCompiler
from src.metrics import Infidelity, TraceDistance
import src.backends.density_matrix.functions as dmf
from src.visualizers.density_matrix import density_matrix_bars
from benchmarks.circuits import *
import src.noise.noise_models as nm


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
        n_stop=100,
    )
    solver.seed(seed)
    solver.solve()

    state = compiler.compile(solver.hof[0][1])

    state = dmf.partial_trace(
        state.data, list(range(n_photon)), (n_photon + n_emitter) * [2]
    )
    return solver.hof, state


def generate_run_no_noise(n_photon, n_emitter, expected_triple, compiler, seed):
    target, _, metric = expected_triple

    solver = EvolutionarySolver(
        target=target,
        metric=metric,
        compiler=compiler,
        n_emitter=n_emitter,
        n_photon=n_photon,
        n_stop=100
    )
    solver.seed(seed)
    solver.solve()

    state = compiler.compile(solver.hof[0][1])

    state = dmf.partial_trace(
        state.data, list(range(n_photon)), (n_photon + n_emitter) * [2]
    )
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
    fig, axs = density_matrix_bars(target_state)
    fig.suptitle("TARGET DENSITY MATRIX")
    plt.show()

    fig, axs = density_matrix_bars(state)
    fig.suptitle("CREATED DENSITY MATRIX")
    plt.show()


def check_run_noise_visual(no_noise_run_info, noise_run_info, expected_info):
    hof_no_noise, state_no_noise = no_noise_run_info
    hof_noise, state_noise = noise_run_info
    target_state, _, metric = expected_info

    circuit_no_noise = hof_no_noise[0][1]
    circuit_no_noise.draw_circuit()

    fig, axs = density_matrix_bars(target_state)
    fig.suptitle("TARGET DENSITY MATRIX")
    plt.show()

    fig, axs = density_matrix_bars(state_no_noise)
    fig.suptitle("CREATED DENSITY MATRIX W/O NOISE")
    plt.show()

    circuit_noise = hof_noise[0][1]
    circuit_noise.draw_circuit()

    fig, axs = density_matrix_bars(state_noise)
    fig.suptitle("CREATED DENSITY MATRIX WITH NOISE")
    plt.show()


@pytest.fixture(scope="module")
def linear3_run_noise(
    density_matrix_compiler, linear3_expected, linear3_noise_model_2
):
    """
    Again, we set the fixture scope to module. Arguably, this is more important than last time because actually
    running the solve takes (relatively) long.

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
    noise_model_mapping = {
        "Identity": nm.OneQubitGateReplacement(
            dmf.parameterized_one_qubit_unitary(np.pi / 180, 0, 0)
        ),
        "SigmaX": nm.DepolarizingNoise(0.01),
        "CNOT": nm.DepolarizingNoise(0.01),
    }
    return noise_model_mapping


@pytest.fixture(scope="module")
def linear3_expected():
    circuit_ideal, state_ideal = linear_cluster_3qubit_circuit()
    target_state = state_ideal["dm"]

    metric = Infidelity(target=target_state)

    return target_state, circuit_ideal, metric


@pytest.fixture(scope="module")
def linear3_expected_trace_distance():
    circuit_ideal, state_ideal = linear_cluster_3qubit_circuit()
    target_state = state_ideal["dm"]
    metric = TraceDistance(target=target_state)
    return target_state, circuit_ideal, metric


@pytest.fixture(scope="module")
def linear4_run_noise(
    density_matrix_compiler, linear4_expected, linear4_noise_model
):
    """
    Again, we set the fixture scope to module. Arguably, this is more important than last time because actually
    running the solve takes (relatively) long.

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
    running the solve takes (relatively) long.

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
    running the solve takes (relatively) long.

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
    circuit_ideal, state_ideal = linear_cluster_4qubit_circuit()
    target_state = state_ideal["dm"]

    metric = Infidelity(target=target_state)

    return target_state, circuit_ideal, metric


@pytest.fixture(scope="module")
def linear4_expected_trace_distance():
    circuit_ideal, state_ideal = linear_cluster_4qubit_circuit()
    target_state = state_ideal["dm"]

    metric = TraceDistance(target=target_state)

    return target_state, circuit_ideal, metric


@pytest.fixture(scope="module")
def ghz3_run_noise(
    density_matrix_compiler, ghz3_expected, ghz3_noise_model
):
    return generate_run_noise(
        3, 1, ghz3_expected, density_matrix_compiler, ghz3_noise_model, 0
    )


@pytest.fixture(scope="module")
def ghz3_run_no_noise(density_matrix_compiler, ghz3_expected):
    return generate_run_no_noise(3, 1, ghz3_expected, density_matrix_compiler, 0)


@pytest.fixture(scope="module")
def ghz3_expected():
    circuit_ideal, state_ideal = ghz3_state_circuit()
    target_state = state_ideal["dm"]

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
    noise_model_mapping = {
        "Hadamard": nm.DepolarizingNoise(0.1),
        "MeasurementCNOTandRest": nm.DepolarizingNoise(0.1),
        "CNOT": nm.DepolarizingNoise(0.1),
        "Identity": nm.DepolarizingNoise(0.1),
    }
    return noise_model_mapping


@visualization
def test_solver_linear3_visualized(
    linear3_run_no_noise, linear3_run_noise, linear3_expected
):
    check_run_noise_visual(linear3_run_no_noise, linear3_run_noise, linear3_expected)


@pytest.fixture(scope="module")
def linear4_noise_model():
    noise_model_mapping = {
        "Hadamard": nm.DepolarizingNoise(0.1),
        "MeasurementCNOTandRest": nm.DepolarizingNoise(0.1),
        "Phase": nm.DepolarizingNoise(0.1),
        "CNOT": nm.DepolarizingNoise(0.1),
        "Identity": nm.DepolarizingNoise(0.1),
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
    noise_model_mapping = {
        "Phase": nm.PhasePerturbedError(0, 0, np.pi / 180),
        "SigmaX": nm.DepolarizingNoise(0.01),
        "CNOT": nm.DepolarizingNoise(0.01),
        "Identity": nm.PauliError("X"),
    }
    return noise_model_mapping


@visualization
def test_solver_ghz3_visualized(ghz3_run_no_noise, ghz3_run_noise, ghz3_expected):
    check_run_noise_visual(ghz3_run_no_noise, ghz3_run_noise, ghz3_expected)
