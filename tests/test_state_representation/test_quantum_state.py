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


def test_convert_representation_graph_to_stabilizer_pure_arm():
    graph = nx.path_graph(3)
    state = QuantumState(graph, rep_type="g")

    state.convert_representation("s")

    expected = rc.graph_to_stabilizer(graph)[0][1]
    assert state.rep_data.data.to_stabilizer() == expected


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
