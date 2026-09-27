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

import graphiq.backends.stabilizer.functions.stabilizer as sfs
from graphiq.backends.density_matrix.compiler import DensityMatrixCompiler
from graphiq.backends.stabilizer.compiler import StabilizerCompiler
from graphiq.backends.stabilizer.state import MixedStabilizer, Stabilizer
from graphiq.circuit.ops import (
    Hadamard,
    Identity,
    MeasurementCNOTandReset,
    MeasurementZ,
    OneQubitGateWrapper,
)
from graphiq.io import IO
from graphiq.metrics import Infidelity
from graphiq.solvers.evolutionary_solver import (
    EvolutionarySolver,
    EvolutionarySolverSetting,
)
from graphiq.visualizers.density_matrix import density_matrix_bars
from tests.test_flags import visualization
from graphiq.benchmarks.circuits import (
    linear_cluster_4qubit_circuit,
    linear_cluster_3qubit_circuit,
    bell_state_circuit,
    ghz3_state_circuit,
)


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
    )
    return solver.hof, state


def check_run(run_info, expected_info):
    hof, state = run_info
    target_state, _, metric = expected_info
    assert np.isclose(hof[0][0], 0.0)  # infidelity score is 0, within numerical error

    circuit = hof[0][1]
    assert np.isclose(hof[0][0], metric.evaluate(state, circuit))
    if state.rep_type == target_state.rep_type == "dm":
        assert np.allclose(state.rep_data.data, target_state.rep_data.data)
    elif state.rep_type == target_state.rep_type == "s":
        assert state.rep_data == target_state.rep_data


def assert_robust_to_measurement_outcome(
    circuit, expected_info, compiler_class, n_photon, n_emitter
):
    """
    Assert that a winning circuit scores 0 infidelity against the target state when every
    measurement is forced to outcome 0 and when every measurement is forced to outcome 1, not
    only under the outcome the solver searched with. For a circuit with a single measurement,
    as every winner certified in this module has, that covers every measurement branch; with
    more than one measurement the mixed outcome patterns are not checked.

    The solver fixtures pin the compiler at measurement_determinism = 1, so every mid-circuit
    measurement in the search, and in check_run, takes that one outcome. A circuit that is 0.0
    there but far from the target in the other branch is effectively post-selecting on a
    measurement outcome. Compiling the circuit under both settings catches it, so a change of
    solver seed cannot silently accept a branch-dependent winner.

    Deliberately not checked under "probabilistic": that draws an outcome at random, so a
    post-selected circuit passes or fails by luck. Forcing both branches is stronger and
    deterministic.

    :param circuit: the circuit to check, e.g. the winner of an evolutionary search
    :type circuit: CircuitDAG
    :param expected_info: (target_state, ideal_circuit, metric) tuple, as produced by the
        ``*_expected`` fixtures in this module
    :type expected_info: tuple
    :param compiler_class: the compiler class to instantiate fresh for each determinism setting
        (e.g. StabilizerCompiler or DensityMatrixCompiler)
    :type compiler_class: type
    :param n_photon: number of photonic qubits in the circuit
    :type n_photon: int
    :param n_emitter: number of emitter qubits in the circuit, traced out before scoring
    :type n_emitter: int
    :return: nothing
    :rtype: None
    """
    target_state, _, metric = expected_info

    # Without a measurement the two compiles are identical and this function guards nothing
    # while still passing.
    assert any(
        isinstance(op, (MeasurementCNOTandReset, MeasurementZ))
        for op in circuit.sequence()
    ), "circuit contains no measurement, so both determinism settings are the same run"

    for determinism in (0, 1):
        compiler = compiler_class()
        compiler.measurement_determinism = determinism
        state = compiler.compile(circuit)
        emitters = list(range(n_photon, n_photon + n_emitter))
        if isinstance(state.rep_data, (Stabilizer, MixedStabilizer)):
            # a stabilizer trace measures the emitters; resolve it in the same branch
            state.rep_data.trace_out_qubits(
                emitters, measurement_determinism=determinism
            )
        else:
            # an exact density-matrix trace has no outcome to resolve
            state.partial_trace(list(range(n_photon)), (n_photon + n_emitter) * [2])
        infidelity = metric.evaluate(state, circuit)
        assert np.isclose(infidelity, 0.0), (
            f"circuit is not robust to measurement outcome: infidelity={infidelity} "
            f"under measurement_determinism={determinism}"
        )


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


@pytest.fixture(scope="module")
def linear3_run(density_matrix_compiler, linear3_expected):
    """
    Again, we set the fixture scope to module. Arguably, this is more important than last time because actually
    running the solve function takes (relatively) long.

    Since we want to apply 2 separate tests on the same run (one visual, one non-visual), it makes sense to have a
    common fixture that only gets called once per module
    """
    return generate_run(3, 1, linear3_expected, density_matrix_compiler, 10)


@pytest.fixture(scope="module")
def linear3_run_stabilizer(stabilizer_compiler, linear3_expected):
    return generate_run(3, 1, linear3_expected, stabilizer_compiler, 10)


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
    return generate_run(4, 1, linear4_expected, density_matrix_compiler, 1)


@pytest.fixture(scope="module")
def linear4_run_stabilizer(stabilizer_compiler, linear4_expected):
    return generate_run(4, 1, linear4_expected, stabilizer_compiler, 1)


@pytest.fixture(scope="module")
def linear4_expected():
    circuit_ideal, state_ideal = linear_cluster_4qubit_circuit()

    metric = Infidelity(target=state_ideal)

    return state_ideal, circuit_ideal, metric


@pytest.fixture(scope="module")
def ghz3_run(density_matrix_compiler, ghz3_expected):
    return generate_run(3, 1, ghz3_expected, density_matrix_compiler, 0)


@pytest.fixture(scope="module")
def ghz3_run_stabilizer(stabilizer_compiler, ghz3_expected):
    return generate_run(3, 1, ghz3_expected, stabilizer_compiler, 0)


@pytest.fixture(scope="module")
def ghz3_expected():
    circuit_ideal, state_ideal = ghz3_state_circuit()

    metric = Infidelity(target=state_ideal)

    return state_ideal, circuit_ideal, metric


def test_solver_linear3(linear3_run, linear3_expected):
    check_run(linear3_run, linear3_expected)
    hof, _ = linear3_run
    assert_robust_to_measurement_outcome(
        hof[0][1], linear3_expected, DensityMatrixCompiler, n_photon=3, n_emitter=1
    )


def test_solver_linear3_stabilizer(linear3_run_stabilizer, linear3_expected):
    check_run(linear3_run_stabilizer, linear3_expected)
    hof, _ = linear3_run_stabilizer
    assert_robust_to_measurement_outcome(
        hof[0][1], linear3_expected, StabilizerCompiler, n_photon=3, n_emitter=1
    )


@visualization
def test_solver_linear3_visualized(linear3_run, linear3_expected):
    check_run_visual(linear3_run, linear3_expected)


def test_solver_linear4(linear4_run, linear4_expected):
    check_run(linear4_run, linear4_expected)
    hof, _ = linear4_run
    assert_robust_to_measurement_outcome(
        hof[0][1], linear4_expected, DensityMatrixCompiler, n_photon=4, n_emitter=1
    )


def test_solver_linear4_stabilizer(linear4_run_stabilizer, linear4_expected):
    check_run(linear4_run_stabilizer, linear4_expected)
    hof, _ = linear4_run_stabilizer
    assert_robust_to_measurement_outcome(
        hof[0][1], linear4_expected, StabilizerCompiler, n_photon=4, n_emitter=1
    )


@visualization
def test_solver_linear4_visualized(linear4_run, linear4_expected):
    check_run_visual(linear4_run, linear4_expected)


def test_solver_ghz3(ghz3_run, ghz3_expected, density_matrix_compiler):
    check_run(ghz3_run, ghz3_expected)
    hof, _ = ghz3_run
    assert_robust_to_measurement_outcome(
        hof[0][1], ghz3_expected, DensityMatrixCompiler, n_photon=3, n_emitter=1
    )


def test_solver_ghz3_stabilizer(
    ghz3_run_stabilizer, ghz3_expected, stabilizer_compiler
):
    check_run(ghz3_run_stabilizer, ghz3_expected)
    hof, _ = ghz3_run_stabilizer
    assert_robust_to_measurement_outcome(
        hof[0][1], ghz3_expected, StabilizerCompiler, n_photon=3, n_emitter=1
    )


@visualization
def test_solver_ghz3_visualized(ghz3_run, ghz3_expected):
    check_run_visual(ghz3_run, ghz3_expected)


def test_assert_robust_to_measurement_outcome_rejects_a_post_selected_winner(
    density_matrix_compiler, ghz3_expected
):
    """
    Seed 25 of the ghz3 density-matrix search ends on a winner that scores 0.0 in the branch the
    compiler is pinned to (measurement_determinism = 1) but is wrong in the other one, so
    check_run accepts it and assert_robust_to_measurement_outcome must not.

    The seed only selects such a winner under the current search path. If a solver change moves
    that path, the precondition below fails; pick a new seed by searching for one whose winner
    scores 0.0 with measurement_determinism = 1 and not with measurement_determinism = 0.
    """
    hof, state = generate_run(3, 1, ghz3_expected, density_matrix_compiler, 25)
    circuit = hof[0][1]
    _, _, metric = ghz3_expected
    branch0_compiler = DensityMatrixCompiler()
    branch0_compiler.measurement_determinism = 0
    branch0_state = branch0_compiler.compile(circuit)
    branch0_state.partial_trace(list(range(3)), 4 * [2])
    assert np.isclose(hof[0][0], 0.0) and not np.isclose(
        metric.evaluate(branch0_state, circuit), 0.0
    ), "seed 25 no longer gives a one-branch winner; pick a new seed (see docstring)"

    check_run((hof, state), ghz3_expected)
    with pytest.raises(AssertionError, match="not robust to measurement outcome"):
        assert_robust_to_measurement_outcome(
            hof[0][1], ghz3_expected, DensityMatrixCompiler, n_photon=3, n_emitter=1
        )


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
        solver_setting=EvolutionarySolverSetting(n_stop=10),
    )
    solver.seed(seed)

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
        solver_setting=EvolutionarySolverSetting(n_stop=5),
    )
    solver.seed(seed)

    solver.solve()
    solver.logs_to_df()


def test_stabilizer_linear3():
    # debugging purpose
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
    solver.seed(10)
    solver.solve()
    state = compiler.compile(solver.hof[0][1])
    state.partial_trace(
        list(range(n_photon)),
        (n_photon + n_emitter) * [2],
    )
    hof = solver.hof
    assert np.isclose(hof[0][0], 0.0)  # infidelity score is 0, within numerical error
    circuit = hof[0][1]
    circuit.draw_circuit()
    assert np.isclose(hof[0][0], metric.evaluate(state, circuit))

    if state.rep_type == target_state.rep_type == "dm":
        assert np.allclose(state.rep_data.data, target_state.rep_data.data)
    elif state.rep_type == target_state.rep_type == "s":
        print(f"the output stabilizer is {state.rep_data.data.to_stabilizer()}")
        output_s_tableau = sfs.canonical_form(state.rep_data.tableau.to_stabilizer())
        print(f"the output stabilizer in the canonical form is {output_s_tableau}")
        target_s_tableau = sfs.canonical_form(
            target_state.rep_data.data.to_stabilizer()
        )
        print(f"the target stabilizer is {target_state.rep_data.data.to_stabilizer()}")
        print(f"the target stabilizer in the canonical form is {target_s_tableau}")
        assert output_s_tableau == target_s_tableau
        assert state.rep_data == target_state.rep_data

    assert_robust_to_measurement_outcome(
        circuit,
        (target_state, circuit_ideal, metric),
        StabilizerCompiler,
        n_photon=n_photon,
        n_emitter=n_emitter,
    )


# Tests for EvolutionarySolver.adapt_probabilities and get_emission_assignment, which have no
# oracle beyond the source and are exercised here directly rather than through a full search.


def test_adapt_probabilities_moves_weight_from_additive_to_removal_transforms():
    solver = EvolutionarySolver(None, None, None, n_emitter=1)
    before = dict(solver.trans_probs)

    solver.adapt_probabilities()

    assert (
        solver.trans_probs[solver.add_emitter_one_qubit_op]
        < before[solver.add_emitter_one_qubit_op]
    )
    assert (
        solver.trans_probs[solver.replace_photon_one_qubit_op]
        < before[solver.replace_photon_one_qubit_op]
    )
    assert solver.trans_probs[solver.remove_op] > before[solver.remove_op]
    assert solver.add_emitter_cnot not in solver.trans_probs
    assert np.isclose(sum(solver.trans_probs.values()), 1)


def test_adapt_probabilities_also_shrinks_the_cnot_key_with_multiple_emitters():
    solver = EvolutionarySolver(None, None, None, n_emitter=2)
    before = dict(solver.trans_probs)

    solver.adapt_probabilities()

    assert solver.trans_probs[solver.add_emitter_cnot] < before[solver.add_emitter_cnot]


def test_adapt_probabilities_adapts_only_the_allowed_transformations():
    solver = EvolutionarySolver(None, None, None, n_emitter=1)
    solver.set_allowed_transformations(
        {solver.add_emitter_one_qubit_op: 0.5, solver.remove_op: 0.5}
    )
    before = dict(solver.trans_probs)

    solver.adapt_probabilities()

    assert set(solver.trans_probs) == set(before)
    assert (
        solver.trans_probs[solver.add_emitter_one_qubit_op]
        < before[solver.add_emitter_one_qubit_op]
    )
    assert solver.trans_probs[solver.remove_op] > before[solver.remove_op]
    assert np.isclose(sum(solver.trans_probs.values()), 1)


def test_adapt_probabilities_without_the_cnot_transformation():
    solver = EvolutionarySolver(None, None, None, n_emitter=2)
    solver.set_allowed_transformations(
        {
            solver.add_emitter_one_qubit_op: 1 / 3,
            solver.replace_photon_one_qubit_op: 1 / 3,
            solver.remove_op: 1 / 3,
        }
    )
    before = dict(solver.trans_probs)

    solver.adapt_probabilities()

    assert set(solver.trans_probs) == set(before)
    for key in (solver.add_emitter_one_qubit_op, solver.replace_photon_one_qubit_op):
        assert solver.trans_probs[key] < before[key]
    assert solver.trans_probs[solver.remove_op] > before[solver.remove_op]
    assert np.isclose(sum(solver.trans_probs.values()), 1)


def test_adapt_probabilities_without_the_emitter_one_qubit_or_remove_transformations():
    solver = EvolutionarySolver(None, None, None, n_emitter=1)
    solver.set_allowed_transformations(
        {solver.replace_photon_one_qubit_op: 0.5, solver.add_emitter_cnot: 0.5}
    )
    before = dict(solver.trans_probs)

    solver.adapt_probabilities()

    assert set(solver.trans_probs) == set(before)
    assert (
        solver.trans_probs[solver.replace_photon_one_qubit_op]
        < before[solver.replace_photon_one_qubit_op]
    )
    assert np.isclose(sum(solver.trans_probs.values()), 1)


def test_adapt_probabilities_leaves_the_cnot_key_on_a_single_emitter_solver():
    # only the normalisation moves the cnot entry on a one-emitter solver, so its ratio
    # to remove_op is the one remove_op's own update gives
    solver = EvolutionarySolver(None, None, None, n_emitter=1)
    solver.set_allowed_transformations(
        {
            solver.add_emitter_one_qubit_op: 0.25,
            solver.replace_photon_one_qubit_op: 0.25,
            solver.remove_op: 0.25,
            solver.add_emitter_cnot: 0.25,
        }
    )
    before = dict(solver.trans_probs)

    solver.adapt_probabilities()

    after = solver.trans_probs
    new_remove = min(before[solver.remove_op] + 1 / solver.setting.n_stop, 0.99)
    assert np.isclose(
        after[solver.add_emitter_cnot] / after[solver.remove_op],
        before[solver.add_emitter_cnot] / new_remove,
    )


def test_adapt_probabilities_stay_non_strict_monotone_over_many_calls():
    # The clamp to [0.01, 0.99] happens before normalisation, so it is not observable in the raw
    # values -- normalised, every decreasing key plateaus at a fixed point (0.0099 for one
    # emitter, with remove_op at 0.9802) rather than continuing to shrink, so strict
    # monotonicity would eventually fail. Non-strict (<=, >=) holds throughout.
    solver = EvolutionarySolver(None, None, None, n_emitter=1)
    n_stop = solver.setting.n_stop
    decreasing_keys = [
        solver.add_emitter_one_qubit_op,
        solver.replace_photon_one_qubit_op,
    ]

    previous = dict(solver.trans_probs)
    for _ in range(3 * n_stop + 1):
        solver.adapt_probabilities()
        for key in decreasing_keys:
            assert solver.trans_probs[key] <= previous[key]
        assert solver.trans_probs[solver.remove_op] >= previous[solver.remove_op]
        assert np.isclose(sum(solver.trans_probs.values()), 1)
        previous = dict(solver.trans_probs)

    assert max(solver.trans_probs, key=solver.trans_probs.get) == solver.remove_op


@pytest.mark.parametrize("n_photon, n_emitter", [(4, 2), (6, 3), (3, 3)])
def test_get_emission_assignment_invariants_hold_over_many_seeded_draws(
    n_photon, n_emitter
):
    # Keep (6, 3): it is the only case where the reuse assertion at the end is load-bearing.
    # With the random-branch growth removed, (4, 2) still reuses emitter 1 by chance, while
    # (6, 3) can only ever reach emitter 2 through the forced tail, exactly once.
    # Do not add a case with n_photon < n_emitter here: every invariant below fails for it
    # (measured for (2, 3) and (3, 5)), which is a separate question about input validation
    # that this function does not perform.
    np.random.seed(2026)
    top_emitter_reused = False
    for _ in range(2000):
        assignment = EvolutionarySolver.get_emission_assignment(n_photon, n_emitter)
        top_emitter_reused |= assignment.count(n_emitter - 1) > 1

        assert len(assignment) == n_photon
        assert assignment[0] == 0
        assert set(assignment) == set(range(n_emitter))

        # Contiguous: a new emitter id, once it first appears, is always exactly one more than
        # the largest id seen so far. This is also why an index into available_emitter can
        # double as the emitter id it names: available_emitter is maintained as exactly
        # list(range(len(available_emitter))) throughout the assignment loop.
        largest_seen = 0
        for emitter_id in assignment:
            assert emitter_id <= largest_seen + 1
            largest_seen = max(largest_seen, emitter_id)

    # The invariants above are all enforced by the forced tail branch alone. The random branch
    # is what lets an emitter be reused after it first appears; without it the top emitter
    # could only ever appear once, in the forced tail. (3, 3) has no slack, so skip it there.
    if n_photon > n_emitter:
        assert top_emitter_reused


# Tests for replace_photon_one_qubit_op on the photon gate that initialization places after each
# emission and labels "Fixed" so that remove_op leaves it in place. With one photon and one
# emitter it is the only photonic one-qubit gate, so the replace always picks it.


def _one_photon_circuit():
    solver = EvolutionarySolver(None, None, None, n_emitter=1, n_photon=1)
    circuit = solver.initialization([0], [0])
    (node,) = circuit.get_node_by_labels(["OneQubitGateWrapper", "Photonic"])
    return solver, circuit, node


def test_replacing_the_fixed_photon_gate_keeps_it_fixed():
    solver, circuit, node = _one_photon_circuit()

    solver.replace_photon_one_qubit_op(circuit)
    solver.remove_op(circuit, node)

    assert node in circuit.dag.nodes
    assert "Fixed" in circuit.dag.nodes[node]["op"].labels


def test_replacing_an_unfixed_photon_gate_leaves_it_removable():
    solver, circuit, node = _one_photon_circuit()
    # stands in for a gate a mutation added, which carries no "Fixed" label
    circuit.replace_op(
        node, OneQubitGateWrapper([Identity, Hadamard], register=0, reg_type="p")
    )

    solver.replace_photon_one_qubit_op(circuit)

    assert "Fixed" not in circuit.dag.nodes[node]["op"].labels
    solver.remove_op(circuit, node)
    assert node not in circuit.dag.nodes


def _saved_files(folder):
    if not folder.exists():
        return set()
    return {path.name for path in folder.iterdir()}


@pytest.mark.parametrize("save_openqasm", ["hof", "both"])
def test_saving_the_hall_of_fame_skips_unfilled_slots(tmp_path, save_openqasm):
    # two candidates per iteration fill 2, 4, then all 5 slots of the hall of fame;
    # a slot no candidate has filled yet has no circuit to save
    _, target = linear_cluster_3qubit_circuit()
    setting = EvolutionarySolverSetting(
        n_stop=3, n_pop=2, n_hof=5, save_openqasm=save_openqasm
    )
    solver = EvolutionarySolver(
        target=target,
        metric=Infidelity(target),
        compiler=DensityMatrixCompiler(),
        n_emitter=1,
        n_photon=3,
        solver_setting=setting,
        io=IO(path=tmp_path, verbose=False),
    )
    EvolutionarySolver.seed(1)

    solver.solve()

    filled = {0: 2, 1: 4, 2: 5}
    assert _saved_files(tmp_path / "hof") == {
        f"iteration{k}_hof{i}.txt" for k, n in filled.items() for i in range(n)
    }
    expected_pop = (
        {f"iteration{k}_pop{j}.txt" for k in range(3) for j in range(2)}
        if save_openqasm == "both"
        else set()
    )
    assert _saved_files(tmp_path / "pop") == expected_pop
