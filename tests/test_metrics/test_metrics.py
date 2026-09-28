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
"""
Tests for graphiq/metrics.py: the circuit-structure metrics, GraphMetric, the
Metrics wrapper, the Infidelity branch that converts a non-stabilizer state into
a (possibly mixed) stabilizer representation, and TraceDistance.
"""

import networkx as nx
import numpy as np
import pytest

import graphiq.backends.density_matrix.functions as dmf
import graphiq.circuit.ops as ops
import graphiq.noise.noise_models as nm
from graphiq.backends.density_matrix.compiler import DensityMatrixCompiler
from graphiq.backends.stabilizer.compiler import StabilizerCompiler
from graphiq.backends.stabilizer.functions.rep_conversion import (
    get_clifford_tableau_from_graph,
)
from graphiq.backends.state_rep_conversion import graph_to_density
from graphiq.benchmarks.alternate_circuits import noise_model_pure_loss
from graphiq.circuit.circuit_dag import CircuitDAG
from graphiq.solvers.alternate_target_solver import graph_to_circ
from graphiq.state import QuantumState
from graphiq.metrics import (
    CircuitCnotCount,
    CircuitDepth,
    CircuitEmitterCount,
    CircuitMaxEmitDepth,
    CircuitMaxEmitEffDepth,
    CircuitMaxEmitResetDepth,
    CircuitMeasureCount,
    CircuitUnitaryCount,
    GraphMetric,
    Infidelity,
    Metrics,
    TraceDistance,
    graph_met_value,
)


def _emitter_circuit(n_emitter, n_photon=0, n_classical=0):
    return CircuitDAG(n_emitter=n_emitter, n_photon=n_photon, n_classical=n_classical)


def test_circuit_depth_default_penalty_returns_depth():
    circuit = _emitter_circuit(1)
    circuit.add(ops.Hadamard(register=0, reg_type="e"))
    circuit.add(ops.Hadamard(register=0, reg_type="e"))
    assert CircuitDepth().evaluate(None, circuit) == circuit.depth


def test_circuit_depth_custom_penalty_applied():
    circuit = _emitter_circuit(1)
    circuit.add(ops.Hadamard(register=0, reg_type="e"))
    metric = CircuitDepth(depth_penalty=lambda d: d + 100)
    assert metric.evaluate(None, circuit) == circuit.depth + 100


def test_circuit_emitter_count_default_penalty_returns_count():
    circuit = _emitter_circuit(3)
    assert CircuitEmitterCount().evaluate(None, circuit) == 3


def test_circuit_emitter_count_custom_penalty_applied():
    circuit = _emitter_circuit(3)
    metric = CircuitEmitterCount(n_emitter_penalty=lambda n: 10 * n)
    assert metric.evaluate(None, circuit) == 30


# The following CircuitCnotCount tests are regression coverage for the
# CircuitCnotCount.__init__ bug, where the default-penalty branch set
# self.n_emitter_penalty instead of self.n_cnot_penalty, so evaluate() raised
# AttributeError whenever n_cnot_penalty was left at its default.


def test_circuit_cnot_count_no_cnots_default_penalty():
    circuit = _emitter_circuit(2)
    circuit.add(ops.Hadamard(register=0, reg_type="e"))
    assert CircuitCnotCount().evaluate(None, circuit) == 0


def test_circuit_cnot_count_counts_cnots_with_default_penalty():
    circuit = _emitter_circuit(2)
    circuit.add(ops.CNOT(control=0, control_type="e", target=1, target_type="e"))
    circuit.add(ops.CNOT(control=1, control_type="e", target=0, target_type="e"))
    assert CircuitCnotCount().evaluate(None, circuit) == 2


def test_circuit_cnot_count_custom_penalty_applied():
    circuit = _emitter_circuit(2)
    circuit.add(ops.CNOT(control=0, control_type="e", target=1, target_type="e"))
    metric = CircuitCnotCount(n_cnot_penalty=lambda n: 10 * n)
    assert metric.evaluate(None, circuit) == 10


# The following CircuitUnitaryCount tests are regression coverage for the
# CircuitUnitaryCount bug, where the gate label list checked "SigmaX" three
# times instead of "SigmaX", "SigmaY", "SigmaZ", so SigmaY/SigmaZ gates were
# never counted.


def test_circuit_unitary_count_counts_sigma_y_and_z():
    circuit = _emitter_circuit(1)
    circuit.add(ops.SigmaY(register=0, reg_type="e"))
    circuit.add(ops.SigmaZ(register=0, reg_type="e"))
    circuit.add(ops.Hadamard(register=0, reg_type="e"))
    circuit.add(ops.Phase(register=0, reg_type="e"))
    circuit.add(ops.PhaseDagger(register=0, reg_type="e"))
    # SigmaY, SigmaZ, Hadamard, Phase, PhaseDagger: 5 unitaries, no SigmaX
    # present at all. The pre-fix code only ever checked "SigmaX" (thrice) so
    # it would report 3 (Hadamard + Phase + PhaseDagger) here.
    assert CircuitUnitaryCount().evaluate(None, circuit) == 5


def test_circuit_unitary_count_counts_all_gate_types_once_each():
    circuit = _emitter_circuit(2)
    circuit.add(ops.SigmaX(register=0, reg_type="e"))
    circuit.add(ops.SigmaY(register=0, reg_type="e"))
    circuit.add(ops.SigmaZ(register=0, reg_type="e"))
    circuit.add(ops.Hadamard(register=0, reg_type="e"))
    circuit.add(ops.Phase(register=0, reg_type="e"))
    circuit.add(ops.PhaseDagger(register=0, reg_type="e"))
    circuit.add(ops.CNOT(control=0, control_type="e", target=1, target_type="e"))
    assert CircuitUnitaryCount().evaluate(None, circuit) == 7


def test_circuit_measure_count_counts_measurement_and_reset_ops():
    circuit = _emitter_circuit(1, n_photon=2, n_classical=1)
    circuit.add(
        ops.MeasurementCNOTandReset(
            control=0, control_type="e", target=0, target_type="p", c_register=0
        )
    )
    circuit.add(
        ops.MeasurementCNOTandReset(
            control=0, control_type="e", target=1, target_type="p", c_register=0
        )
    )
    assert CircuitMeasureCount().evaluate(None, circuit) == 2


def test_circuit_measure_count_custom_penalty_applied():
    circuit = _emitter_circuit(1, n_photon=1, n_classical=1)
    circuit.add(
        ops.MeasurementCNOTandReset(
            control=0, control_type="e", target=0, target_type="p", c_register=0
        )
    )
    metric = CircuitMeasureCount(m_penalty=lambda n: n + 5)
    assert metric.evaluate(None, circuit) == 6


# CircuitMaxEmitDepth, CircuitMaxEmitResetDepth, and CircuitMaxEmitEffDepth
# rely on undocumented CircuitDAG internals (reg_gate_history, _max_depth).
# The expected values below were derived by constructing each circuit and printing
# reg_gate_history/_max_depth/the metric's own output directly, rather than
# by hand-tracing the DAG, to avoid the test itself encoding a wrong
# understanding of these internals.


def test_circuit_max_emit_depth_default_penalty_is_max_across_emitters():
    circuit = _emitter_circuit(2)
    circuit.add(ops.Hadamard(register=0, reg_type="e"))
    circuit.add(ops.Hadamard(register=1, reg_type="e"))
    circuit.add(ops.Hadamard(register=1, reg_type="e"))
    circuit.add(ops.Hadamard(register=1, reg_type="e"))
    # emitter 0 has 1 gate node, emitter 1 has 3 -> max is 3
    assert CircuitMaxEmitDepth().evaluate(None, circuit) == 3


def test_circuit_max_emit_depth_no_gates_is_zero():
    circuit = _emitter_circuit(1)
    assert CircuitMaxEmitDepth().evaluate(None, circuit) == 0


def test_circuit_max_emit_depth_custom_penalty_applied():
    circuit = _emitter_circuit(2)
    circuit.add(ops.Hadamard(register=0, reg_type="e"))
    circuit.add(ops.Hadamard(register=1, reg_type="e"))
    circuit.add(ops.Hadamard(register=1, reg_type="e"))
    circuit.add(ops.Hadamard(register=1, reg_type="e"))
    metric = CircuitMaxEmitDepth(depth_penalty=lambda d: d + 100)
    assert metric.evaluate(None, circuit) == 103


def _reset_circuit():
    # A single emitter measured-and-reset twice, with photons/classical
    # register interleaved: Input, H, H, M&R, H, M&R, Output. The longest
    # gap between consecutive Input/M&R/Output markers is 3 nodes (Input to
    # the first M&R).
    circuit = _emitter_circuit(1, n_photon=2, n_classical=1)
    circuit.add(ops.Hadamard(register=0, reg_type="e"))
    circuit.add(ops.Hadamard(register=0, reg_type="e"))
    circuit.add(
        ops.MeasurementCNOTandReset(
            control=0, control_type="e", target=0, target_type="p", c_register=0
        )
    )
    circuit.add(ops.Hadamard(register=0, reg_type="e"))
    circuit.add(
        ops.MeasurementCNOTandReset(
            control=0, control_type="e", target=1, target_type="p", c_register=0
        )
    )
    return circuit


def test_circuit_max_emit_reset_depth_default_penalty():
    circuit = _reset_circuit()
    assert CircuitMaxEmitResetDepth().evaluate(None, circuit) == 3


def test_circuit_max_emit_reset_depth_custom_penalty_applied():
    circuit = _reset_circuit()
    metric = CircuitMaxEmitResetDepth(depth_penalty=lambda d: d + 100)
    assert metric.evaluate(None, circuit) == 103


def test_circuit_max_emit_eff_depth_matches_reset_depth_without_branching():
    # With no two-qubit gates, DAG depth along a single emitter's own
    # chronological chain increases by exactly 1 per node, so
    # CircuitMaxEmitEffDepth reduces to the same value as
    # CircuitMaxEmitResetDepth on this circuit.
    circuit = _reset_circuit()
    assert CircuitMaxEmitEffDepth().evaluate(None, circuit) == 3


def test_circuit_max_emit_eff_depth_custom_penalty_applied():
    circuit = _reset_circuit()
    metric = CircuitMaxEmitEffDepth(depth_penalty=lambda d: d + 100)
    assert metric.evaluate(None, circuit) == 103


def test_circuit_max_emit_eff_depth_diverges_from_reset_depth_with_cross_register_depth():
    # CircuitMaxEmitResetDepth counts nodes along each emitter's own chain,
    # blind to what CNOTs bring in from other registers; CircuitMaxEmitEffDepth
    # uses the DAG's true topological depth (_max_depth), which does pick up
    # that cross-register depth. Chain e2 (5 Hadamards) -> CNOT into e1 (short
    # chain) -> CNOT into e0 (short chain, then 2 measure-and-resets): e2's own
    # depth injects into e1 and then e0, inflating e0/e1's effective depth well
    # past their own node counts, while e2's own long chain still dominates the
    # node-count-based CircuitMaxEmitResetDepth.
    circuit = _emitter_circuit(3, n_photon=2, n_classical=1)
    for _ in range(5):
        circuit.add(ops.Hadamard(register=2, reg_type="e"))
    circuit.add(ops.Hadamard(register=1, reg_type="e"))
    circuit.add(ops.CNOT(control=2, control_type="e", target=1, target_type="e"))
    circuit.add(ops.Hadamard(register=0, reg_type="e"))
    circuit.add(ops.CNOT(control=1, control_type="e", target=0, target_type="e"))
    circuit.add(
        ops.MeasurementCNOTandReset(
            control=0, control_type="e", target=0, target_type="p", c_register=0
        )
    )
    circuit.add(ops.Hadamard(register=0, reg_type="e"))
    circuit.add(
        ops.MeasurementCNOTandReset(
            control=0, control_type="e", target=1, target_type="p", c_register=0
        )
    )

    assert CircuitMaxEmitResetDepth().evaluate(None, circuit) == 7
    assert CircuitMaxEmitEffDepth().evaluate(None, circuit) == 8


def _circuit_without_emitters():
    circuit = _emitter_circuit(0, n_photon=2)
    circuit.add(ops.Hadamard(register=0, reg_type="p"))
    circuit.add(ops.CNOT(control=0, control_type="p", target=1, target_type="p"))
    return circuit


_EMITTER_DEPTH_METRICS = [
    CircuitMaxEmitDepth,
    CircuitMaxEmitResetDepth,
    CircuitMaxEmitEffDepth,
]


@pytest.mark.parametrize("metric_class", _EMITTER_DEPTH_METRICS)
def test_emitter_depth_metrics_on_a_circuit_without_emitters(metric_class):
    assert metric_class().evaluate(None, _circuit_without_emitters()) == 0


def test_emitter_depth_metrics_by_name_on_a_circuit_without_emitters():
    metrics = Metrics([m.__name__ for m in _EMITTER_DEPTH_METRICS])
    assert metrics.evaluate(None, _circuit_without_emitters()) == 0.0


@pytest.mark.parametrize("metric_class", _EMITTER_DEPTH_METRICS)
def test_emitter_depth_penalty_applies_without_emitters(metric_class):
    metric = metric_class(depth_penalty=lambda d: d + 100)
    assert metric.evaluate(None, _circuit_without_emitters()) == 100
    assert metric.log == [100]


@pytest.fixture
def path_graph():
    # 0 - 1 - 2 - 3, a simple, hand-checkable graph
    return nx.path_graph(4)


def test_graph_met_value_node_and_edge_counts(path_graph):
    assert graph_met_value("node", path_graph) == 4
    assert graph_met_value("n_edges", path_graph) == 3


def test_graph_met_value_max_degree(path_graph):
    # both middle nodes (1 and 2) have degree 2
    assert graph_met_value("max_deg", path_graph) == 2


def test_graph_met_value_radius_and_diameter(path_graph):
    assert graph_met_value("radius", path_graph) == 2
    assert graph_met_value("diameter", path_graph) == 3


def test_graph_met_value_pop_is_edge_density(path_graph):
    # density of a path graph on 4 nodes: 3 edges / C(4, 2) = 3 / 6 = 0.5
    assert graph_met_value("pop", path_graph) == pytest.approx(0.5)


def test_graph_met_value_unknown_metric_raises(path_graph):
    with pytest.raises(ValueError):
        graph_met_value("not_a_real_metric", path_graph)


def test_graph_metric_evaluate_default_penalty(path_graph):
    metric = GraphMetric(path_graph)
    assert metric.evaluate("node") == 4


def test_graph_metric_evaluate_custom_penalty(path_graph):
    metric = GraphMetric(path_graph, penalty=lambda x: x * 2)
    assert metric.evaluate("n_edges") == 6


# The following Metrics-wrapper tests stick to CircuitDepth, with distinct
# penalty functions to get distinguishable per-metric values.


def test_metrics_default_equal_weighting():
    circuit = _emitter_circuit(1)
    circuit.add(ops.Hadamard(register=0, reg_type="e"))
    depth = circuit.depth
    metrics = Metrics(
        [
            CircuitDepth(depth_penalty=lambda d: d),
            CircuitDepth(depth_penalty=lambda d: d + 1),
        ]
    )
    assert metrics.evaluate(None, circuit) == depth + (depth + 1)


def test_metrics_list_weighting():
    circuit = _emitter_circuit(1)
    circuit.add(ops.Hadamard(register=0, reg_type="e"))
    depth = circuit.depth
    metrics = Metrics(
        [
            CircuitDepth(depth_penalty=lambda d: d),
            CircuitDepth(depth_penalty=lambda d: d + 1),
        ],
        metric_weight=[1, 10],
    )
    assert metrics.evaluate(None, circuit) == 1 * depth + 10 * (depth + 1)


def test_metrics_callable_weighting():
    circuit = _emitter_circuit(1)
    metrics = Metrics(
        [CircuitDepth()],
        metric_weight=lambda state, circuit: 99,
    )
    assert metrics.evaluate(None, circuit) == 99


@pytest.mark.parametrize("metric_weight", [(1, 10), 2.0])
def test_metrics_rejects_an_unsupported_weight_type(metric_weight):
    # a tuple or a scalar used to be accepted, and the first evaluate then
    # raised AttributeError for the missing weighting function
    with pytest.raises(TypeError, match="metric_weight"):
        Metrics([CircuitDepth(), CircuitEmitterCount()], metric_weight=metric_weight)


def test_metrics_per_metric_log():
    circuit = _emitter_circuit(1)
    circuit.add(ops.Hadamard(register=0, reg_type="e"))
    metrics = Metrics([CircuitDepth()])
    metrics.evaluate(None, circuit)
    log = metrics.per_metric_log
    assert log["CircuitDepth"] == [circuit.depth]


def test_metrics_string_metric_name():
    circuit = _emitter_circuit(1)
    circuit.add(ops.Hadamard(register=0, reg_type="e"))
    metrics = Metrics(["CircuitDepth"])
    assert metrics.evaluate(None, circuit) == circuit.depth


def test_metrics_unrecognized_metric_raises():
    with pytest.raises(UserWarning):
        Metrics(["NotARealMetric"])


@pytest.mark.parametrize(
    "name",
    [
        "CircuitEmitterCount",
        "CircuitCnotCount",
        "CircuitUnitaryCount",
        "CircuitMaxEmitDepth",
        "CircuitMaxEmitResetDepth",
        "CircuitMaxEmitEffDepth",
        "CircuitMeasureCount",
    ],
)
def test_metrics_accepts_each_circuit_metric_by_name(name):
    metrics = Metrics([name])
    assert [type(metric).__name__ for metric in metrics._metrics] == [name]


def test_metrics_rejects_graph_metric_by_name():
    # GraphMetric's evaluate takes a graph-metric name, not (state, circuit)
    with pytest.raises(UserWarning):
        Metrics(["GraphMetric"])


def test_infidelity_graph_state_marked_mixed_matches_stabilizer_target():
    """Regression coverage for the Infidelity.evaluate bug where the mixed-
    stabilizer branch checked `isinstance(state.rep_data, MixedStabilizer)`
    instead of `isinstance(rep_data, MixedStabilizer)`. When the target is
    rep_type "s" but the evaluated state is not, rep_data comes from a
    converted copy, and state.rep_data (the original, un-converted dm/graph
    data) never matches -- so `fid` was left unassigned and evaluate() raised
    UnboundLocalError.

    Builds the |+> state two ways: once as a graph representation (a single
    isolated node, no edges) marked mixed=True -- forcing the "g"->"s"
    conversion this Infidelity branch performs to wrap its result in a
    MixedStabilizer (a single (1.0, tableau) entry) -- and once as a genuine
    Stabilizer state via a Hadamard-only circuit, used as the target. The
    graph->stabilizer conversion is an exact, unambiguous transform, unlike
    the density-matrix route, which goes through an eigendecomposition.
    """
    graph = nx.Graph()
    graph.add_node(0)
    state = QuantumState(data=graph, rep_type="g", mixed=True)

    target_circuit = _emitter_circuit(1)
    target_circuit.add(ops.Hadamard(register=0, reg_type="e"))
    target = StabilizerCompiler().compile(target_circuit)

    infidelity = Infidelity(target=target).evaluate(state, None)

    assert infidelity == pytest.approx(0.0, abs=1e-9)


def test_infidelity_and_trace_distance_of_a_graph_state_marked_mixed_against_a_dm_target():
    state = QuantumState(nx.path_graph(3), rep_type="g", mixed=True)
    target = QuantumState(graph_to_density(nx.path_graph(3)), rep_type="dm")

    assert Infidelity(target).evaluate(state, None) == pytest.approx(0.0, abs=1e-9)
    assert TraceDistance(target).evaluate(state, None) == pytest.approx(0.0, abs=1e-9)


def _empty_plus_triangle_mixture():
    # two orthogonal 3-qubit graph states with distinct weights
    return 0.7 * graph_to_density(nx.empty_graph(3)) + 0.3 * graph_to_density(
        nx.complete_graph(3)
    )


def test_infidelity_of_a_mixed_density_matrix_against_a_stabilizer_target():
    rho = _empty_plus_triangle_mixture()
    state = QuantumState(rho, rep_type="dm", mixed=True)
    target = QuantumState(
        get_clifford_tableau_from_graph(nx.empty_graph(3)), rep_type="s"
    )

    infidelity = Infidelity(target=target).evaluate(state, None)

    expected = 1 - dmf.fidelity(graph_to_density(nx.empty_graph(3)), rho)
    assert infidelity == pytest.approx(expected)


def test_infidelity_rejects_a_mixed_density_matrix_not_marked_mixed():
    # Converting it would keep one branch of the mixture and score only that branch.
    state = QuantumState(_empty_plus_triangle_mixture(), rep_type="dm")
    target = QuantumState(
        get_clifford_tableau_from_graph(nx.empty_graph(3)), rep_type="s"
    )

    with pytest.raises(ValueError, match="mixed=True"):
        Infidelity(target=target).evaluate(state, None)


def test_infidelity_rejects_a_lossy_density_matrix_not_marked_mixed():
    # Converting it to a stabilizer state would drop the lost 0.1 and score it lossless.
    state = QuantumState(graph_to_density(nx.path_graph(3)), rep_type="dm")
    state.rep_data.apply_channel([np.sqrt(0.9) * np.eye(8)])
    assert np.real(np.trace(state.rep_data.data)) == pytest.approx(0.9)
    target = QuantumState(
        get_clifford_tableau_from_graph(nx.path_graph(3)), rep_type="s"
    )

    with pytest.raises(ValueError, match="mixed=True"):
        Infidelity(target=target).evaluate(state, None)


def test_trace_distance_stabilizer_state_against_its_own_density_matrix_is_zero():
    # TraceDistance.evaluate only compares density matrices; a stabilizer-represented state is
    # converted to "dm" on the fly (a copy, so the original state is left untouched).
    state = QuantumState(1, rep_type="s")
    target = QuantumState(1, rep_type="s")
    target.convert_representation("dm")

    trace_distance = TraceDistance(target=target).evaluate(state, None)

    assert trace_distance == pytest.approx(0.0, abs=1e-9)


def test_trace_distance_stabilizer_ket0_against_density_matrix_ket1_is_one():
    state = QuantumState(1, rep_type="s")  # |0>
    target = QuantumState(dmf.ket2dm(dmf.state_ketz1()), rep_type="dm")  # |1>

    trace_distance = TraceDistance(target=target).evaluate(state, None)

    assert trace_distance == pytest.approx(1.0, abs=1e-9)


def test_trace_distance_raises_when_target_is_not_a_density_matrix():
    # The check is keyed on the target's rep_type, not the evaluated state's.
    state = QuantumState(1, rep_type="s")
    target = QuantumState(1, rep_type="s")

    with pytest.raises(ValueError, match="Cannot compute the trace distance"):
        TraceDistance(target=target).evaluate(state, None)


def _path3_mixture():
    return 0.5 * graph_to_density(nx.path_graph(3)) + 0.5 * graph_to_density(
        nx.star_graph(2)
    )


@pytest.mark.parametrize(
    "rho", [graph_to_density(nx.path_graph(3)), _path3_mixture()], ids=["pure", "mixed"]
)
def test_infidelity_counts_the_missing_trace_of_a_lossy_density_matrix(rho):
    target = QuantumState(rho, rep_type="dm")
    state = QuantumState(rho, rep_type="dm")
    # lose 0.1 in place; building the state from 0.9 * rho would renormalise it
    state.rep_data.apply_channel([np.sqrt(0.9) * np.eye(8)])

    assert Infidelity(target=target).evaluate(state, None) == pytest.approx(0.1)


def test_infidelity_of_a_density_matrix_with_trace_one_up_to_rounding_is_unchanged():
    target = QuantumState(graph_to_density(nx.path_graph(3)), rep_type="dm")
    state = QuantumState(graph_to_density(nx.path_graph(3)), rep_type="dm")
    # four gate and dephasing steps leave the trace at 1 - 1.4e-15
    h = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    unitaries = [np.kron(np.eye(2), np.kron(h, h)), np.kron(h, np.eye(4))]
    dephasing = [np.sqrt(0.9) * np.eye(8), np.sqrt(0.1) * np.diag([1, -1] * 4)]
    for step in range(4):
        state.rep_data.apply_unitary(unitaries[step % 2])
        state.rep_data.apply_channel(dephasing)
    sigma, rho = target.rep_data.data, state.rep_data.data
    trace = np.real(np.trace(rho))
    assert np.trace(rho) != 1.0 and np.isclose(trace, 1)
    # the lossy-state formula rounds differently here, so the check can tell
    assert 1 - trace * dmf.fidelity(sigma, rho / trace) != 1 - dmf.fidelity(sigma, rho)

    infidelity = Infidelity(target=target).evaluate(state, None)

    assert infidelity == 1 - dmf.fidelity(sigma, rho)


def _lossy_linear4(compiler, determinism, loss=0.1, lossy_gate_after=False):
    circuit = graph_to_circ(nx.path_graph(4)).assign_noise(noise_model_pure_loss(loss))
    if lossy_gate_after:
        # a loss after photon 3's last measurement, then a gate undoing the
        # lossy Hadamard so the state still overlaps the target
        circuit.add(ops.Hadamard(register=3, reg_type="p", noise=nm.PhotonLoss(0.1)))
        circuit.add(ops.Hadamard(register=3, reg_type="p"))
    compiler.noise_simulation = True
    compiler.measurement_determinism = determinism
    state = compiler.compile(circuit)
    state.partial_trace(keep=list(range(4)), dims=(4 + circuit.n_emitters) * [2])
    return state


def _stabilizer_target_linear4():
    return QuantumState(get_clifford_tableau_from_graph(nx.path_graph(4)), rep_type="s")


def _density_matrix_target_linear4():
    return QuantumState(graph_to_density(nx.path_graph(4)), rep_type="dm")


@pytest.mark.parametrize("determinism", [0, 1])
def test_infidelity_of_a_lossy_density_matrix_against_a_stabilizer_target(determinism):
    state = _lossy_linear4(DensityMatrixCompiler(), determinism)

    infidelity = Infidelity(target=_stabilizer_target_linear4()).evaluate(state, None)

    assert infidelity == pytest.approx(0.1)


@pytest.mark.parametrize("determinism", [0, 1])
@pytest.mark.parametrize("loss", [0.1, 1e-6])
def test_density_matrix_infidelity_under_loss_matches_the_stabilizer_one(
    loss, determinism
):
    dm_state = _lossy_linear4(DensityMatrixCompiler(), determinism, loss)
    stab_state = _lossy_linear4(StabilizerCompiler(), determinism, loss)

    dm = Infidelity(target=_density_matrix_target_linear4()).evaluate(dm_state, None)
    stab = Infidelity(target=_stabilizer_target_linear4()).evaluate(stab_state, None)

    assert stab == pytest.approx(loss, rel=1e-6)
    # relative only: an absolute tolerance near 1e-6 would accept 0.0
    assert dm == pytest.approx(stab, rel=1e-6)


@pytest.mark.parametrize("determinism", [0, 1])
def test_density_matrix_infidelity_with_a_loss_after_the_last_measurement(determinism):
    dm_state = _lossy_linear4(
        DensityMatrixCompiler(), determinism, lossy_gate_after=True
    )
    stab_state = _lossy_linear4(
        StabilizerCompiler(), determinism, lossy_gate_after=True
    )

    dm = Infidelity(target=_density_matrix_target_linear4()).evaluate(dm_state, None)
    stab = Infidelity(target=_stabilizer_target_linear4()).evaluate(stab_state, None)

    assert stab == pytest.approx(0.19)
    assert dm == pytest.approx(stab)


_MIXED_RHO = np.array([[0.5, -0.2j], [0.2j, 0.5]])
_MIXED_SIGMA = np.array([[0.6, 0.1 - 0.1j], [0.1 + 0.1j, 0.4]])
# for qubits, F = tr(rho sigma) + 2 sqrt(det rho det sigma)
_F_MIXED = 0.54 + 2 * np.sqrt(0.21 * 0.22)


@pytest.mark.parametrize("loss", [0.0, 0.1])
def test_infidelity_of_a_mixed_density_matrix_against_a_mixed_target(loss):
    target = QuantumState(_MIXED_RHO, rep_type="dm")
    state = QuantumState(_MIXED_SIGMA, rep_type="dm")
    if loss:
        # in place; building the state from (1 - loss) * sigma would renormalise it
        state.rep_data.apply_channel([np.sqrt(1 - loss) * np.eye(2)])

    infidelity = Infidelity(target=target).evaluate(state, None)

    assert infidelity == pytest.approx(1 - (1 - loss) * _F_MIXED)
