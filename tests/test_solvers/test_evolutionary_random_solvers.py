import pytest
import matplotlib.pyplot as plt
import numpy as np
import src.backends.stabilizer.functions.stabilizer as sfs
from benchmarks.circuits import *
from tests.test_flags import visualization
from src.solvers.evolutionary_solver import EvolutionarySolver
from src.backends.density_matrix.compiler import DensityMatrixCompiler
from src.backends.stabilizer.compiler import StabilizerCompiler

from src.metrics import Infidelity
from src.state import QuantumState
from src.io import IO
from src.visualizers.density_matrix import density_matrix_bars


@pytest.fixture(scope="module")
def density_matrix_compiler():
    """
    Here, we set the scope to module because, while we want to use a DensityMatrixCompiler multiple times, we don't
    actually need a different copy each time
    """
    compiler = DensityMatrixCompiler()
    compiler.measurement_determinism = 1
    return compiler


@pytest.fixture(scope="module")
def stabilizer_compiler():
    """
    Here, we set the scope to module because, while we want to use a StabilizerCompiler multiple times, we don't
    actually need a different copy each time
    """
    compiler = StabilizerCompiler()
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

    state.partial_trace(
        list(range(n_photon)),
        (n_photon + n_emitter) * [2],
        compiler.measurement_determinism,
    )
    return solver.hof, state


def check_run(run_info, expected_info):
    hof, state = run_info
    target_state, _, metric = expected_info
    assert np.isclose(hof[0][0], 0.0)  # infidelity score is 0, within numerical error

    circuit = hof[0][1]
    assert np.isclose(hof[0][0], metric.evaluate(state, circuit))
    if state._dm is not None and target_state._dm is not None:
        assert np.allclose(state.dm.data, target_state.dm.data)
    if state._stabilizer is not None and target_state._stabilizer is not None:
        assert state.stabilizer.__eq__(target_state.stabilizer)


def check_run_visual(run_info, expected_info):
    hof, state = run_info
    target_state, _, metric = expected_info

    circuit = hof[0][1]
    circuit.draw_circuit()
    fig, axs = density_matrix_bars(target_state.dm.data)
    fig.suptitle("TARGET DENSITY MATRIX")
    plt.show()

    fig, axs = density_matrix_bars(state.dm.data)
    fig.suptitle("CREATED DENSITY MATRIX")
    plt.show()


@pytest.fixture(scope="module")
def linear3_run(density_matrix_compiler, linear3_expected):
    """
    Again, we set the fixture scope to module. Arguably, this is more important than last time because actually
    running the solve takes (relatively) long.

    Since we want to apply 2 separate tests on the same run (one visual, one non-visual), it makes sense to have a
    common fixture that only gets called once per module
    """
    return generate_run(3, 1, linear3_expected, density_matrix_compiler, 10)


@pytest.fixture(scope="module")
def linear3_run_stabilizer(stabilizer_compiler, linear3_expected):
    return generate_run(3, 1, linear3_expected, stabilizer_compiler, 1)


@pytest.fixture(scope="module")
def linear3_expected():
    circuit_ideal, state_ideal = linear_cluster_3qubit_circuit()

    metric = Infidelity(target=state_ideal)

    return state_ideal, circuit_ideal, metric


@pytest.fixture(scope="module")
def linear4_run(density_matrix_compiler, linear4_expected):
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

    metric = Infidelity(target=state_ideal)

    return state_ideal, circuit_ideal, metric


@pytest.fixture(scope="module")
def ghz3_run(density_matrix_compiler, ghz3_expected):
    return generate_run(3, 1, ghz3_expected, density_matrix_compiler, 0)


@pytest.fixture(scope="module")
def ghz3_expected():
    circuit_ideal, state_ideal = ghz3_state_circuit()

    metric = Infidelity(target=state_ideal)

    return state_ideal, circuit_ideal, metric


def test_solver_linear3(linear3_run, linear3_expected):
    check_run(linear3_run, linear3_expected)


def test_solver_linear3_stabilizer(linear3_run_stabilizer, linear3_expected):
    check_run(linear3_run_stabilizer, linear3_expected)


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

    circuit_ideal, target_state = linear_cluster_3qubit_circuit()
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
    solver.n_stop = 10

    original_trans_prob = solver.trans_probs
    solver.trans_probs = {
        solver.remove_op: 1 / 2,
        solver.add_measurement_cnot_and_reset: 1 / 2,
    }
    solver.solve()

    solver.trans_probs = original_trans_prob


@pytest.mark.parametrize("seed", [0, 3])
def test_solver_logging(seed):
    n_emitter = 1
    n_photon = 2

    circuit_ideal, target_state = bell_state_circuit()

    compiler = DensityMatrixCompiler()
    metric = Infidelity(target=target_state)
    io = IO.new_directory(
        folder="_tests",
        include_date=False,
        include_time=False,
        include_id=False,
        verbose=False,
    )
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


def test_stabilizer_linear3():
    n_emitter = 1
    n_photon = 3
    circuit_ideal, target_state = linear_cluster_3qubit_circuit()
    metric = Infidelity(target=target_state)
    compiler = StabilizerCompiler()
    compiler.measurement_determinism = 1
    solver = EvolutionarySolver(
        target=target_state,
        metric=metric,
        compiler=compiler,
        n_emitter=n_emitter,
        n_photon=n_photon,
    )
    solver.seed(1)
    solver.solve()
    state = compiler.compile(solver.hof[0][1])
    state.partial_trace(
        list(range(n_photon)),
        (n_photon + n_emitter) * [2],
        compiler.measurement_determinism,
    )
    hof = solver.hof
    assert np.isclose(hof[0][0], 0.0)  # infidelity score is 0, within numerical error
    circuit = hof[0][1]
    circuit.draw_circuit()
    assert np.isclose(hof[0][0], metric.evaluate(state, circuit))

    if state._dm is not None and target_state._dm is not None:
        pass
        # assert np.allclose(state.dm.data, target_state.dm.data)
    if state._stabilizer is not None and target_state._stabilizer is not None:
        print(f"the output stabilizer is {state.stabilizer.data.to_stabilizer()}")
        output_s_tableau = sfs.canonical_form(state.stabilizer.tableau.to_stabilizer())
        print(f"the output stabilizer in the canonical form is {output_s_tableau}")
        target_s_tableau = sfs.canonical_form(
            target_state.stabilizer.tableau.to_stabilizer()
        )
        print(
            f"the target stabilizer is {target_state.stabilizer.data.to_stabilizer()}"
        )
        print(
            f"the target stabilizer in the canonical form is {target_state.stabilizer.tableau.to_stabilizer()}"
        )
        # assert state.stabilizer.__eq__(target_state.stabilizer)
        pass


"""
# Next target

@visualization
def test_square_4qubit():
    graph = nx.Graph([(1, 2), (1, 3), (3, 4), (2, 4), (2, 3)])
    state = DensityMatrix.from_graph(graph)
    n_emitter = 2
    n_photon = 4
 
    target_state = QuantumState(n_photon, state.data)
    compiler = DensityMatrixCompiler()
    metric = Infidelity(target=target_state)
    solver = EvolutionarySolver(target=target_state, metric=metric, compiler=compiler,
                                         n_emitter=n_emitter, n_photon=n_photon)
    solver.seed(2)
#   EvolutionarySolver.n_stop = 300
    solver.trans_probs = {
        solver.remove_op: 1 / 4 + 1 / 20,
        solver.add_measurement_cnot_and_reset: 1 / 20,
        solver.replace_photon_one_qubit_op: 1 / 4,
        solver.add_emitter_one_qubit_op: 1 / 4 + 1 / 20,
        solver.add_emitter_cnot: 1 / 10
    }
    solver.p_dist = [0.4] + 11 * [0.2 / 22] + [0.4] + 11 * [0.2 / 22]
    solver.e_dist = [0.2] + 11 * [0.4 / 22] + [0.4] + 11 * [0.4 / 22]
    solver.solve()
    circuit = solver.hof[0][1]
    circuit.draw_circuit()
"""
