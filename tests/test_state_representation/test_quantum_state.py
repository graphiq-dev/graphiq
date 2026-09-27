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
import matplotlib.axes
import matplotlib.figure
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pytest

from graphiq.backends.density_matrix import numpy as dmnp
import graphiq.backends.state_rep_conversion as rc
from graphiq.backends.graph.state import Graph, MixedGraph
from graphiq.backends.stabilizer.clifford_tableau import CliffordTableau
from graphiq.backends.stabilizer.functions.clifford import create_n_ket0_state
from graphiq.backends.stabilizer.functions.transformation import run_circuit
from graphiq.backends.stabilizer.state import MixedStabilizer
from graphiq.state import DENSITY_MATRIX_QUBIT_THRESH, QuantumState


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
def test_initializing_dm_1(n):
    data = dmnp.eye(2**n)
    state = QuantumState(data, rep_type="dm")
    assert np.allclose(state.rep_data.data, data / np.trace(data))


def test_initializing_dm_2():
    """Test that it fails if we don't provide the right datatype"""
    data = "bad input"
    with pytest.raises(TypeError):
        QuantumState(data, rep_type="dm")


def test_initializing_graph_2():
    n = 3
    data = nx.Graph([(1, 2), (2, 3)])
    state = QuantumState(data, rep_type="g")
    assert len(state.rep_data.data.nodes) == n


def test_initializing_stabilizer_1():
    """Checks that we can initialize stabilizers with integers"""
    n = 130
    state = QuantumState(n, rep_type="s")
    assert state.rep_data.n_qubits == n


def test_initializing_stabilizer_2():
    """Checks that we can initialize stabilizers with a CliffordTableau"""
    n = 130
    state = QuantumState(CliffordTableau(n), rep_type="s")
    assert state.rep_data.n_qubits == n


def test_convert_representation_graph_stabilizer_graph_round_trip_pure_arm():
    graph = nx.path_graph(3)
    state = QuantumState(graph, rep_type="g")

    state.convert_representation("s")
    state.convert_representation("g")

    assert nx.is_isomorphic(state.rep_data.data, graph)


def test_convert_representation_stabilizer_to_graph_of_a_circuit_prepared_graph_state():
    # the star graph state (centre 0) as a circuit leaves it: generators XZZ, ZXI, IXX
    tableau = run_circuit(
        CliffordTableau(3),
        [("H", 0), ("CNOT", 0, 1), ("CNOT", 1, 2), ("H", 1), ("H", 2)],
    )
    state = QuantumState(tableau, rep_type="s")

    state.convert_representation("g")

    assert sorted(state.rep_data.data.edges) == [(0, 1), (0, 2)]


def _edge_set(graph):
    return {frozenset(edge) for edge in graph.edges}


@pytest.mark.parametrize(
    "edges", [[(0, 1), (1, 2)], [(0, 1)]], ids=["path3", "isolated_node"]
)
def test_convert_representation_density_matrix_to_graph_pure_arm(edges):
    graph = nx.Graph(edges)
    graph.add_nodes_from(range(3))
    rho = rc.graph_to_density(graph)
    state = QuantumState(rho, rep_type="dm")

    state.convert_representation("g")

    assert isinstance(state.rep_data, Graph)
    assert set(state.rep_data.data.nodes) == {0, 1, 2}
    assert _edge_set(state.rep_data.data) == _edge_set(graph)
    assert all(not data for _, _, data in state.rep_data.data.edges(data=True))
    state.convert_representation("dm")
    assert np.allclose(state.rep_data.data, rho)


def test_convert_representation_density_matrix_to_graph_pure_arm_marked_mixed():
    rho = rc.graph_to_density(nx.path_graph(3))
    state = QuantumState(rho, rep_type="dm", mixed=True)

    state.convert_representation("g")

    assert isinstance(state.rep_data, MixedGraph)
    assert len(state.rep_data.mixture) == 1
    weight, graph = state.rep_data.mixture[0]
    assert weight == 1.0
    assert _edge_set(graph.data) == _edge_set(nx.path_graph(3))
    state.convert_representation("dm")
    assert np.allclose(state.rep_data.data, rho)


def _empty_plus_triangle_mixture():
    # two orthogonal graph states with distinct weights
    return 0.7 * rc.graph_to_density(nx.empty_graph(3)) + 0.3 * rc.graph_to_density(
        nx.complete_graph(3)
    )


def test_convert_representation_density_matrix_to_graph_mixed_arm():
    rho = _empty_plus_triangle_mixture()
    state = QuantumState(rho, rep_type="dm", mixed=True)

    state.convert_representation("g")

    assert isinstance(state.rep_data, MixedGraph)
    # the six zero-weight eigenvectors are not branches of the mixture
    assert len(state.rep_data.mixture) == 2
    branches = sorted(
        (round(float(p_i), 9), sorted(map(sorted, _edge_set(g_i.data))))
        for p_i, g_i in state.rep_data.mixture
    )
    assert branches == [(0.3, [[0, 1], [0, 2], [1, 2]]), (0.7, [])]
    for _, g_i in state.rep_data.mixture:
        assert set(g_i.data.nodes) == {0, 1, 2}
    state.convert_representation("dm")
    assert np.allclose(state.rep_data.data, rho)


def test_convert_representation_density_matrix_to_graph_rejects_a_mixture_not_marked_mixed():
    state = QuantumState(_empty_plus_triangle_mixture(), rep_type="dm")

    with pytest.raises(ValueError, match="mixed=True"):
        state.convert_representation("g")


def test_convert_representation_stabilizer_to_graph_mixed_arm():
    path3 = nx.path_graph(3)
    star3 = nx.star_graph(2)
    mixture = [
        (0.7, CliffordTableau(rc.graph_to_stabilizer(path3)[0][1])),
        (0.3, CliffordTableau(rc.graph_to_stabilizer(star3)[0][1])),
    ]
    state = QuantumState(mixture, rep_type="s", mixed=True)

    state.convert_representation("g")

    mixed_graph = state.rep_data
    assert isinstance(mixed_graph, MixedGraph)
    assert [p_i for p_i, _ in mixed_graph.mixture] == [0.7, 0.3]
    assert nx.is_isomorphic(mixed_graph.mixture[0][1].data, path3)
    assert nx.is_isomorphic(mixed_graph.mixture[1][1].data, star3)


def test_str_of_a_mixed_graph():
    # MixedGraph never sets _data, so the base class __str__ must go through data
    mixed_graph = MixedGraph([(1.0, Graph(nx.path_graph(2)))])
    assert str(mixed_graph).startswith("MixedGraph")


def test_graph_to_density_mixed_arm_reached_through_the_private_helper():
    # This calls the private helper directly on an instance whose "mixed" flag is set, since
    # the constructor rejects a MixedGraph. The same arm is also reached publicly by converting
    # a mixed stabilizer state to "g" and then to "dm".
    qs = QuantumState(nx.path_graph(3), rep_type="g", mixed=True)
    graph_a = nx.path_graph(3)
    graph_b = nx.cycle_graph(3)
    mixed_graph = MixedGraph([(0.7, Graph(graph_a)), (0.3, Graph(graph_b))])

    density = qs._graph_to_density(mixed_graph)

    expected = 0.7 * rc.graph_to_density(graph_a) + 0.3 * rc.graph_to_density(graph_b)
    assert np.allclose(density.data, expected)


def test_convert_representation_graph_to_density_matrix_pure_arm():
    graph = nx.path_graph(3)
    state = QuantumState(graph, rep_type="g")

    state.convert_representation("dm")

    assert np.allclose(state.rep_data.data, rc.graph_to_density(graph))


def test_convert_representation_graph_marked_mixed_to_density_matrix():
    # a state built from a graph with mixed=True holds a plain Graph, not a MixedGraph
    graph = nx.path_graph(3)
    state = QuantumState(graph, rep_type="g", mixed=True)

    state.convert_representation("dm")

    assert state.rep_type == "dm"
    assert np.allclose(state.rep_data.data, rc.graph_to_density(graph))


def test_convert_representation_graph_to_stabilizer_pure_arm():
    graph = nx.path_graph(3)
    state = QuantumState(graph, rep_type="g")

    state.convert_representation("s")

    expected = rc.graph_to_stabilizer(graph)[0][1]
    assert state.rep_data.data.to_stabilizer() == expected


def _graph_tableau(graph):
    return CliffordTableau(rc.graph_to_stabilizer(graph)[0][1])


def test_convert_representation_mixed_stabilizer_to_graph_and_back():
    path3 = nx.path_graph(3)
    star3 = nx.star_graph(2)
    mixture = [(0.7, _graph_tableau(path3)), (0.3, _graph_tableau(star3))]
    state = QuantumState(mixture, rep_type="s", mixed=True)

    state.convert_representation("g")
    state.convert_representation("s")

    mixed_stabilizer = state.rep_data
    assert isinstance(mixed_stabilizer, MixedStabilizer)
    assert [p_i for p_i, _ in mixed_stabilizer.mixture] == [0.7, 0.3]
    assert mixed_stabilizer.mixture[0][1] == _graph_tableau(path3)
    assert mixed_stabilizer.mixture[1][1] == _graph_tableau(star3)
    # the two tableaux differ, so the comparisons above can fail
    assert not mixed_stabilizer.mixture[0][1] == _graph_tableau(star3)


def test_convert_representation_density_matrix_marked_mixed_to_graph_to_stabilizer():
    path3 = nx.path_graph(3)
    state = QuantumState(rc.graph_to_density(path3), rep_type="dm", mixed=True)

    state.convert_representation("g")
    state.convert_representation("s")

    mixed_stabilizer = state.rep_data
    assert isinstance(mixed_stabilizer, MixedStabilizer)
    assert len(mixed_stabilizer.mixture) == 1
    assert np.isclose(mixed_stabilizer.mixture[0][0], 1.0)
    assert mixed_stabilizer.mixture[0][1] == _graph_tableau(path3)


def test_convert_representation_plain_graph_marked_mixed_to_stabilizer():
    # a state built from a graph with mixed=True holds a plain Graph, not a MixedGraph
    graph = nx.path_graph(3)
    state = QuantumState(graph, rep_type="g", mixed=True)

    state.convert_representation("s")

    mixed_stabilizer = state.rep_data
    assert isinstance(mixed_stabilizer, MixedStabilizer)
    assert [p_i for p_i, _ in mixed_stabilizer.mixture] == [1.0]
    assert mixed_stabilizer.mixture[0][1] == _graph_tableau(graph)


def test_graph_to_stabilizer_rejects_a_mixed_graph_on_a_state_not_marked_mixed():
    # This calls the private helper directly, since the constructor rejects a MixedGraph.
    qs = QuantumState(nx.path_graph(3), rep_type="g")

    with pytest.raises(TypeError):
        qs._graph_to_stabilizer(MixedGraph(Graph(nx.path_graph(3))))


@pytest.mark.parametrize("rep_type", ["g", None])
def test_partial_trace_on_a_graph_state_says_it_is_not_implemented(rep_type):
    # rep_type=None autodetects the graph representation; it is refused like an
    # explicit one rather than failing with an AttributeError.
    state = QuantumState(nx.path_graph(4), rep_type=rep_type)

    with pytest.raises(NotImplementedError, match="Graph representation"):
        state.partial_trace(keep=[0], dims=4 * [2])


@pytest.mark.parametrize(
    "data, expected_rep_type",
    [
        (np.eye(4) / 4, "dm"),
        (2, "dm"),
        (create_n_ket0_state(3), "s"),
        (DENSITY_MATRIX_QUBIT_THRESH, "s"),
        (nx.path_graph(3), "g"),
    ],
)
def test_autodetected_representation_records_its_rep_type(data, expected_rep_type):
    # Without an explicit rep_type, __init__ picks the representation from the data;
    # the name must be recorded too, since everything downstream dispatches on it.
    state = QuantumState(data)

    assert state.rep_type == expected_rep_type


def test_autodetected_state_converts_like_an_explicit_one():
    graph = nx.path_graph(3)
    autodetected = QuantumState(graph)
    explicit = QuantumState(graph, rep_type="g")

    autodetected.convert_representation("dm")
    explicit.convert_representation("dm")

    assert autodetected.rep_type == "dm"
    assert np.allclose(autodetected.rep_data.data, explicit.rep_data.data)


def test_show_on_a_graph_state_returns_the_figure_and_axes():
    # show() unpacks the (fig, ax) that Graph.draw returns. No ax is passed, so
    # this covers the ax=None path: the current figure and its current axes.
    state = QuantumState(nx.path_graph(3), rep_type="g")
    fresh = plt.figure()
    try:
        fig, ax = state.show(show=False)

        assert isinstance(fig, matplotlib.figure.Figure)
        assert isinstance(ax, matplotlib.axes.Axes)
        assert fig is fresh
        assert ax.collections  # the nodes and edges nx.draw added
    finally:
        plt.close(fresh)


def test_show_on_a_graph_state_returns_the_axes_it_was_given():
    state = QuantumState(nx.path_graph(3), rep_type="g")
    f1, (a1, a2) = plt.subplots(1, 2)
    f2 = plt.figure()  # the current figure and axes are now neither f1 nor a1
    try:
        fig, ax = state.show(show=False, ax=a1)

        assert fig is f1
        assert ax is a1
        assert a1.collections
        assert not a2.collections
    finally:
        plt.close(f1)
        plt.close(f2)


def test_show_on_a_graph_state_in_a_subfigure_returns_the_root_figure():
    state = QuantumState(nx.path_graph(3), rep_type="g")
    root = plt.figure()
    sub_ax = root.subfigures(1, 2)[0].subplots()
    other = plt.figure()
    try:
        fig, ax = state.show(show=False, ax=sub_ax)

        assert isinstance(fig, matplotlib.figure.Figure)
        assert fig is root
        assert ax is sub_ax
    finally:
        plt.close(root)
        plt.close(other)


def test_show_on_a_mixed_graph_state_raises_not_implemented():
    t1 = QuantumState(nx.path_graph(3), rep_type="g")
    t1.convert_representation("s")
    t2 = QuantumState(nx.star_graph(2), rep_type="g")
    t2.convert_representation("s")
    state = QuantumState(
        [(0.5, t1.rep_data.tableau), (0.5, t2.rep_data.tableau)],
        rep_type="s",
        mixed=True,
    )
    state.convert_representation("g")
    assert isinstance(state.rep_data, MixedGraph)

    with pytest.raises(NotImplementedError, match="mixed graph state"):
        state.show(show=False)


def _nx_graph_mixture():
    return [(0.5, nx.path_graph(3)), (0.5, nx.star_graph(2))]


def test_validate_data_rejects_a_list_of_networkx_graphs():
    # No representation here can be built from a list of (p, nx.Graph), so
    # validate_data reports it as unsupported rather than reading n_qubits,
    # which an nx.Graph does not have.
    assert QuantumState.validate_data(_nx_graph_mixture()) == (False, None)


def test_constructor_rejects_a_list_of_networkx_graphs_with_type_error():
    with pytest.raises(TypeError, match="Data's type is incorrect"):
        QuantumState(_nx_graph_mixture(), rep_type="g", mixed=True)


def test_validate_data_accepts_a_list_of_clifford_tableaux():
    assert QuantumState.validate_data([(1.0, CliffordTableau(3))]) == (True, 3)


# QuantumState.n_qubits used to be stored once in __init__ and never updated, so a traced or
# otherwise resized state kept reporting its old width while the representation reported the
# real one. It is now derived from the representation, and the rep_data setter checks the
# incoming data against that live width.


def _traced_state(rep_type, mixed=False):
    """A 4-qubit state traced down to its first 3 qubits."""
    state = QuantumState(4, rep_type=rep_type, mixed=mixed)
    state.partial_trace(keep=[0, 1, 2], dims=4 * [2])
    return state


@pytest.mark.parametrize("rep_type", ["dm", "s"])
def test_n_qubits_follows_a_partial_trace(rep_type):
    assert _traced_state(rep_type).n_qubits == 3


def test_n_qubits_follows_a_partial_trace_on_a_mixed_stabilizer():
    # Reaches the count through mixture[0][1], a different path from the pure case.
    assert _traced_state("s", mixed=True).n_qubits == 3


def test_n_qubits_follows_a_resize_made_through_the_representation():
    # rep_data hands out the live representation, so resizing it bypasses QuantumState. A count
    # updated only in partial_trace would still go stale here.
    state = QuantumState(4, rep_type="s")

    state.rep_data.remove_qubit(0)

    assert state.n_qubits == 3


@pytest.mark.parametrize(
    "rep_type, mixed",
    [("dm", False), ("s", False), ("s", True)],
    ids=["dm", "s", "mixed s"],
)
def test_a_traced_state_accepts_its_own_data_back(rep_type, mixed):
    # Raw data, not the representation object, so the assignment is rebuilt and goes through
    # the width check, which used to compare it against the pre-trace count.
    state = _traced_state(rep_type, mixed=mixed)
    rep = state.rep_data

    state.rep_data = rep.mixture if mixed else rep.data

    assert state.n_qubits == 3


@pytest.mark.parametrize("rep_type", ["dm", "s"])
def test_a_traced_state_rejects_data_of_its_old_width(rep_type):
    state = _traced_state(rep_type)

    with pytest.raises(AssertionError):
        state.rep_data = np.eye(16, dtype=complex) / 16 if rep_type == "dm" else 4


@pytest.mark.parametrize(
    "n_nodes, accepted", [(4, True), (3, False)], ids=["new width", "old width"]
)
def test_a_graph_state_grown_in_place_checks_against_its_new_width(n_nodes, accepted):
    state = QuantumState(nx.path_graph(3), rep_type="g")
    state.rep_data.add_node(3)

    if accepted:
        state.rep_data = nx.path_graph(n_nodes)
    else:
        with pytest.raises(AssertionError):
            state.rep_data = nx.path_graph(n_nodes)

    assert state.n_qubits == 4


def test_assigning_a_wider_density_matrix_is_rejected():
    state = QuantumState(np.eye(8, dtype=complex) / 8, rep_type="dm")

    with pytest.raises(AssertionError):
        state.rep_data = np.eye(32, dtype=complex) / 32

    assert state.n_qubits == 3


def test_n_qubits_is_read_only():
    state = QuantumState(3, rep_type="s")

    with pytest.raises(AttributeError):
        state.n_qubits = 5


@pytest.mark.parametrize(
    "assign", [False, True], ids=["at construction", "through the setter"]
)
def test_a_non_power_of_two_density_matrix_is_rejected(assign):
    bad = np.eye(9, dtype=complex) / 9

    if assign:
        state = QuantumState(np.eye(8, dtype=complex) / 8, rep_type="dm")
        with pytest.raises(AssertionError):
            state.rep_data = bad
    else:
        with pytest.raises(AssertionError):
            QuantumState(bad, rep_type="dm")


def _emitter_circuit(n_emitter, hadamards):
    from graphiq.circuit.circuit_dag import CircuitDAG
    import graphiq.circuit.ops as ops

    circuit = CircuitDAG(n_emitter=n_emitter, n_photon=0, n_classical=0)
    for register in hadamards:
        circuit.add(ops.Hadamard(register=register, reg_type="e"))
    return circuit


def test_a_traced_state_can_seed_a_circuit_of_its_new_width():
    # CompilerBase asserts initial_state.n_qubits == circuit.n_quantum, so a stale count rejected
    # a state whose data was exactly the circuit's width. The Hadamard applied before the trace
    # shows the compiled state really started from initial_state.
    from graphiq.backends.stabilizer.compiler import StabilizerCompiler

    state = QuantumState(4, rep_type="s")
    state.rep_data.apply_hadamard(0)
    state.partial_trace(keep=[0, 1, 2], dims=4 * [2])

    compiled = StabilizerCompiler().compile(
        _emitter_circuit(3, [2]), initial_state=state
    )
    expected = StabilizerCompiler().compile(_emitter_circuit(3, [0, 2]))

    assert compiled.n_qubits == 3
    assert compiled.rep_data.data == expected.rep_data.data


def test_a_traced_state_cannot_seed_a_circuit_of_its_old_width():
    from graphiq.backends.stabilizer.compiler import StabilizerCompiler

    with pytest.raises(AssertionError, match="must match the circuit"):
        StabilizerCompiler().compile(
            _emitter_circuit(4, []), initial_state=_traced_state("s")
        )
