import pytest
from tests.test_flags import visualization

import matplotlib.pyplot as plt

from benchmarks.circuits import *

from src.solvers.evolutionary_solver import EvolutionarySolver
from src.backends.density_matrix.compiler import DensityMatrixCompiler
import src.backends.density_matrix.functions as dmf
from src.metrics import Infidelity
from src.io import IO

from src.visualizers.density_matrix import density_matrix_bars


@pytest.fixture(scope="module")
def solver_stop_100():
    """
    Fixtures like these can be used to set up and teardown preparations for a test.

    Here, the first two lines will run BEFORE the test, the test will run after the yield, and the lines
    after the yield will run AFTER the test
    """
    n_stop_original = EvolutionarySolver.n_stop
    EvolutionarySolver.n_stop = 100
    yield
    EvolutionarySolver.n_stop = n_stop_original


@pytest.fixture(scope="module")
def density_matrix_compiler():
    """
    Here, we set the scope to module because, while we want to use a DensityMatrixCompiler multiple times, we don't
    actually need a different copy each time
    """
    compiler = DensityMatrixCompiler()
    compiler.measurement_determinism = 1
    return compiler


def generate_run(n_photon, n_emitter, expected_triple, compiler, seed):
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

    state = dmf.partial_trace(
        state.data, list(range(n_photon)), (n_photon + n_emitter) * [2]
    )
    return solver.hof, state


def check_run(run_info, expected_info):
    hof, state = run_info
    target_state, _, metric = expected_info
    assert np.isclose(hof[0][0], 0.0)  # infidelity score is 0, within numerical error

    circuit = hof[0][1]
    assert np.isclose(hof[0][0], metric.evaluate(state, circuit))
    assert np.allclose(state, target_state)


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


@pytest.fixture(scope="module")
def linear3_run(solver_stop_100, density_matrix_compiler, linear3_expected):
    """
    Again, we set the fixture scope to module. Arguably, this is more important than last time because actually
    running the solve takes (relatively) long.

    Since we want to apply 2 separate tests on the same run (one visual, one non-visual), it makes sense to have a
    common fixture that only gets called once per module
    """
    return generate_run(3, 1, linear3_expected, density_matrix_compiler, 10)


@pytest.fixture(scope="module")
def linear3_expected():
    circuit_ideal, state_ideal = linear_cluster_3qubit_circuit()
    target_state = state_ideal["dm"]

    metric = Infidelity(target=target_state)

    return target_state, circuit_ideal, metric


@pytest.fixture(scope="module")
def linear4_run(solver_stop_100, density_matrix_compiler, linear4_expected):
    """
    Again, we set the fixture scope to module. Arguably, this is more important than last time because actually
    running the solve takes (relatively) long.

    Since we want to apply 2 separate tests on the same run (one visual, one non-visual), it makes sense to have a
    common fixture that only gets called once per module
    """
    return generate_run(4, 1, linear4_expected, density_matrix_compiler, 10)


@pytest.fixture(scope="module")
def linear4_expected():
    circuit_ideal, state_ideal = linear_cluster_4qubit_circuit()
    target_state = state_ideal["dm"]

    metric = Infidelity(target=target_state)

    return target_state, circuit_ideal, metric


@pytest.fixture(scope="module")
def ghz3_run(solver_stop_100, density_matrix_compiler, ghz3_expected):
    return generate_run(3, 1, ghz3_expected, density_matrix_compiler, 0)


@pytest.fixture(scope="module")
def ghz3_expected():
    circuit_ideal, state_ideal = ghz3_state_circuit()
    target_state = state_ideal["dm"]

    metric = Infidelity(target=target_state)

    return target_state, circuit_ideal, metric


def test_solver_linear3(linear3_run, linear3_expected):
    check_run(linear3_run, linear3_expected)


@visualization
def test_solver_linear3_visualized(linear3_run, linear3_expected):
    check_run_visual(linear3_run, linear3_expected)


def test_solver_linear4(linear4_run, linear4_expected):
    check_run(linear4_run, linear4_expected)


@visualization
def test_solver_linear4_visualized(linear4_run, linear4_expected):
    check_run_visual(linear4_run, linear4_expected)


@pytest.mark.xfail(
    reason="Took too long to find a good seed (given that we will be updating the code a lot)"
)
def test_solver_ghz3(ghz3_run, ghz3_expected, density_matrix_compiler):
    check_run(ghz3_run, ghz3_expected)


@visualization
def test_solver_ghz3_visualized(ghz3_run, ghz3_expected):
    check_run_visual(ghz3_run, ghz3_expected)


@pytest.mark.parametrize("seed", [0, 3, 325, 2949])
def test_add_remove_measurements(seed):
    n_emitter = 1
    n_photon = 3

    circuit_ideal, state_ideal = linear_cluster_3qubit_circuit()
    target_state = state_ideal["dm"]
    compiler = DensityMatrixCompiler()
    metric = Infidelity(target=target_state)
    solver = EvolutionarySolver(
        target=target_state,
        metric=metric,
        compiler=compiler,
        n_emitter=n_emitter,
        n_photon=n_photon,
    )
    solver.seed(seed)

    original_trans_prob = solver.trans_probs
    solver.trans_probs = {
        solver.remove_op: 1 / 2,
        solver.add_measurement_cnot_and_reset: 1 / 2,
    }
    solver.solve()

    solver.trans_probs = original_trans_prob


@pytest.mark.parametrize("seed", [0, 3, 325, 2949])
def test_solver_logging(seed):
    n_emitter = 1
    n_photon = 2

    circuit_ideal, state_ideal = linear_cluster_3qubit_circuit()
    target_state = state_ideal["dm"]
    compiler = DensityMatrixCompiler()
    metric = Infidelity(target=target_state)
    io = IO.new_directory(folder="_tests", include_date=False, include_time=False, include_id=False, verbose=False)
    solver = EvolutionarySolver(
        target=target_state,
        metric=metric,
        compiler=compiler,
        n_emitter=n_emitter,
        n_photon=n_photon,
        io=io,
    )
    solver.seed(seed)

    solver.solve()
    solver.logs_to_df()
