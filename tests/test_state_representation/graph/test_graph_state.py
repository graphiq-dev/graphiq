import pytest
import matplotlib.pyplot as plt

import src.backends.graph.functions as gfn
from src.backends.graph.state import Graph
from tests.test_flags import visualization


def test_graph_initialization_adding(graph_rep_1):
    assert set(graph_rep_1.get_nodes_id_form()) == {
        frozenset([1]),
        frozenset([2]),
        frozenset([3]),
    }
    assert set(graph_rep_1.get_edges_id_form()) == {
        (frozenset([1]), frozenset([2])),
        (frozenset([2]), frozenset([3])),
    }
    graph_rep_1.add_node(4)
    graph_rep_1.add_edge(3, 4)
    assert set(graph_rep_1.get_edges_id_form()) == {
        (frozenset([1]), frozenset([2])),
        (frozenset([2]), frozenset([3])),
        (frozenset([3]), frozenset([4])),
    }


# @visualization
# def test_graph_draw(graph_rep_1):
#    graph_rep_1.add_node(4)
#    graph_rep_1.add_edge(3, 4)
#    graph_rep_1.draw()


def test_get_node_by_id_no_id(graph_rep_1):
    assert isinstance(graph_rep_1.get_node_by_id(1), gfn.QuNode)
    assert isinstance(graph_rep_1.get_node_by_id(frozenset([1])), gfn.QuNode)

    assert graph_rep_1.get_node_by_id(4) is None
    assert graph_rep_1.get_node_by_id(frozenset([4])) is None


def test_n_redundant_encoding_1(graph_rep_2):
    assert graph_rep_2.n_redundant_encoding_node == 2


def test_n_redundant_encoding_2():
    state_rep = Graph([], root_node_id=None)
    assert state_rep.n_redundant_encoding_node == 0


def test_remove_id(graph_rep_2):
    assert graph_rep_2.node_is_redundant(frozenset([1, 2, 3]))
    graph_rep_2.remove_id_from_redundancy(frozenset([1, 2, 3]), removal_id=1)
    print(graph_rep_2.get_nodes_id_form())
    print(graph_rep_2.get_edges_id_form())
    assert graph_rep_2.node_is_redundant(frozenset([2, 3]))
    graph_rep_2.remove_id_from_redundancy(frozenset([2, 3]), removal_id=2)
    assert not graph_rep_2.node_is_redundant(frozenset([3]))
    graph_rep_2.remove_id_from_redundancy(3)
    assert len(graph_rep_2.data.nodes) == 2


def test_neighbours_1(graph_rep_1):
    """
    Confirm that neighbours are found correctly
    """
    neighbors = graph_rep_1.get_neighbors(2)
    assert len(neighbors) == 2
    neighbor_ids = [n.get_id() for n in neighbors]
    assert frozenset([1]) in neighbor_ids
    assert frozenset([3]) in neighbor_ids


def test_local_complementation_1(graph_rep_1):
    """
    Checks that local complementation can be correctly performed on a graph
    """
    graph_rep_1.local_complementation(
        1
    )  # shouldn't change anything, because only 2 is in the neighborhood
    assert graph_rep_1.edge_exists(1, 2)
    assert graph_rep_1.edge_exists(2, 3)
    assert not graph_rep_1.edge_exists(1, 3)

    graph_rep_1.local_complementation(2)
    assert graph_rep_1.edge_exists(1, 2)
    assert graph_rep_1.edge_exists(2, 3)
    assert graph_rep_1.edge_exists(1, 3)


@visualization
def test_local_complementation_visual_1(graph_rep_1):
    """
    Checks that local complementation can be correctly performed on a graph
    """
    fig, ax = plt.subplots(3, figsize=(10, 10))
    ax[0].set_title("Original graph")
    graph_rep_1.draw(ax=ax[0], show=False)
    graph_rep_1.local_complementation(
        1
    )  # shouldn't change anything, because only 2 is in the neighborhood
    ax[1].set_title("Local complementation on node {1}")
    graph_rep_1.draw(ax=ax[1], show=False)
    graph_rep_1.local_complementation(2)
    ax[2].set_title("Local complementation on node {2}")
    graph_rep_1.draw(ax=ax[2], show=False)
    plt.show()


def test_local_complementation_2(graph_rep_1):
    """
    Checks that local complementation can be correctly performed on a graph
    """
    new_node = frozenset([4, 5, 6])
    graph_rep_1.add_node(new_node)
    graph_rep_1.add_edge(new_node, 2)
    graph_rep_1.local_complementation(
        3
    )  # shouldn't change anything, because only 2 is in the neighborhood
    assert graph_rep_1.edge_exists(1, 2)
    assert graph_rep_1.edge_exists(2, 3)
    assert graph_rep_1.edge_exists(2, new_node)
    assert not graph_rep_1.edge_exists(1, 3)
    assert not graph_rep_1.edge_exists(1, new_node)
    assert not graph_rep_1.edge_exists(3, new_node)

    graph_rep_1.local_complementation(2)
    assert graph_rep_1.edge_exists(1, 2)
    assert graph_rep_1.edge_exists(2, 3)
    assert graph_rep_1.edge_exists(2, new_node)
    assert graph_rep_1.edge_exists(1, 3)
    assert graph_rep_1.edge_exists(1, new_node)
    assert graph_rep_1.edge_exists(3, new_node)


@visualization
def test_local_complementation_visual_2(graph_rep_1):
    """
    Checks that local complementation can be correctly performed on a graph
    """
    graph_rep_1.add_node(frozenset([4, 5, 6]))
    graph_rep_1.add_edge(frozenset([4, 5, 6]), 2)
    fig, ax = plt.subplots(3, figsize=(10, 10))
    ax[0].set_title("Original graph")
    graph_rep_1.draw(ax=ax[0], show=False)
    graph_rep_1.local_complementation(
        3
    )  # shouldn't change anything, because only 2 is in the neighborhood
    ax[1].set_title("Local complementation on node {3}")
    graph_rep_1.draw(ax=ax[1], show=False)
    graph_rep_1.local_complementation(2)
    ax[2].set_title("Local complementation on node {2}")
    graph_rep_1.draw(ax=ax[2], show=False)
    plt.show()


def test_local_complementation_3(graph_rep_1):
    """
    Checks that local complementation can be correctly performed on a graph
    """
    new_node = frozenset([4, 5, 6])
    graph_rep_1.add_node(new_node)
    graph_rep_1.add_edge(new_node, 2)
    graph_rep_1.add_edge(new_node, 1)
    graph_rep_1.local_complementation(
        3
    )  # shouldn't change anything, because only 2 is in the neighborhood
    assert graph_rep_1.edge_exists(1, 2)
    assert graph_rep_1.edge_exists(2, 3)
    assert graph_rep_1.edge_exists(2, new_node)
    assert not graph_rep_1.edge_exists(1, 3)
    assert graph_rep_1.edge_exists(1, new_node)
    assert not graph_rep_1.edge_exists(3, new_node)

    graph_rep_1.local_complementation(2)
    assert graph_rep_1.edge_exists(1, 2)
    assert graph_rep_1.edge_exists(2, 3)
    assert graph_rep_1.edge_exists(2, new_node)
    assert graph_rep_1.edge_exists(1, 3)
    assert not graph_rep_1.edge_exists(1, new_node)
    assert graph_rep_1.edge_exists(3, new_node)


@visualization
def test_local_complementation_visual_3(graph_rep_1):
    """
    Checks that local complementation can be correctly performed on a graph
    """
    graph_rep_1.add_node(frozenset([4, 5, 6]))
    graph_rep_1.add_edge(frozenset([4, 5, 6]), 2)
    graph_rep_1.add_edge(frozenset([4, 5, 6]), 1)
    fig, ax = plt.subplots(3, figsize=(10, 10))
    ax[0].set_title("Original graph")
    graph_rep_1.draw(ax=ax[0], show=False)
    graph_rep_1.local_complementation(
        3
    )  # shouldn't change anything, because only 2 is in the neighborhood
    ax[1].set_title("Local complementation on node {3}")
    graph_rep_1.draw(ax=ax[1], show=False)
    graph_rep_1.local_complementation(2)
    ax[2].set_title("Local complementation on node {2}")
    graph_rep_1.draw(ax=ax[2], show=False)
    plt.show()


def _plot_merge(graph, id1, id2):
    fig, ax = plt.subplots(2, figsize=(10, 7))
    ax[0].set_title("Original graph")
    graph.draw(ax=ax[0], show=False)
    graph.merge(id1, id2)
    ax[1].set_title(f"Merged nodes: {id1}, {id2}")
    graph.draw(ax=ax[1], show=False)
    plt.show()


def test_merge_1(graph_rep_1):
    """
    Check that it will merge two connected nodes
    """
    graph_rep_1.merge(1, 2)
    assert len(graph_rep_1.get_edges()) == 1
    assert graph_rep_1.edge_exists(frozenset([1, 2]), 3)


def test_merge_2(graph_rep_1):
    """
    Check that it will merge two unconnected nodes which have a common neighbour
    """
    graph_rep_1.merge(1, 3)
    assert len(graph_rep_1.get_edges()) == 1
    assert graph_rep_1.edge_exists(frozenset([1, 3]), 2)


def test_merge_3(graph_rep_2):
    """
    Check that we can merge two separate clusters
    """
    graph_rep_2.add_node(8)
    graph_rep_2.add_node(frozenset([9, 10]))
    graph_rep_2.add_node(frozenset([11]))
    graph_rep_2.add_edge(8, 11)
    graph_rep_2.add_edge(8, frozenset([9, 10]))
    graph_rep_2.merge(8, frozenset([1, 2, 3]))
    assert len(graph_rep_2.get_edges()) == 4
    new_node = frozenset([1, 2, 3, 8])
    assert graph_rep_2.edge_exists(new_node, frozenset([9, 10]))
    assert graph_rep_2.edge_exists(new_node, 11)
    assert graph_rep_2.edge_exists(new_node, frozenset([4, 5, 6]))
    assert graph_rep_2.edge_exists(new_node, 7)


@visualization
def test_merge_visual_1(graph_rep_1):
    """
    Check that it will merge two connected nodes
    """
    _plot_merge(graph_rep_1, 1, 2)


@visualization
def test_merge_visual_2(graph_rep_1):
    """
    Check that it will merge two unconnected nodes which have a common neighbour
    """
    _plot_merge(graph_rep_1, 1, 3)


@visualization
def test_merge_visual_3(graph_rep_2):
    """
    Check that we can merge two separate clusters
    """
    graph_rep_2.add_node(8)
    graph_rep_2.add_node(frozenset([9, 10]))
    graph_rep_2.add_node(frozenset([11]))
    graph_rep_2.add_edge(8, 11)
    graph_rep_2.add_edge(8, frozenset([9, 10]))
    _plot_merge(graph_rep_2, 8, frozenset([1, 2, 3]))


@visualization
@pytest.mark.parametrize("graph_choice", [1, 2])
def test_y_measurement_visual_1(graph_choice, graph_rep_1, graph_rep_2):
    """
    Check that y measurement acts as expected on a linear cluster state
    """
    if graph_choice == 1:
        graph = graph_rep_1
        measure = 2
    else:
        graph = graph_rep_2
        measure = frozenset([1, 2, 3])

    fig, ax = plt.subplots(2)
    graph.draw(ax=ax[0], show=False)
    ax[0].set_title("Original")
    graph.measure_y(measure)
    graph.draw(ax=ax[1], show=False)
    ax[1].set_title(f"Y measurement on node {measure}")
    plt.show()


def test_y_measurement_1(graph_rep_1):
    """
    Check that y measurement acts as expected on a linear cluster state
    """
    graph_rep_1.measure_y(2)
    assert len(graph_rep_1.get_edges()) == 1
    assert graph_rep_1.edge_exists(1, 3)
