from src.backends.graph.functions import QuNode
from src.backends.graph.state import Graph
from tests.test_flags import visualization


def test_graph_initialization_adding(graph_rep_1):
    assert set(graph_rep_1.get_nodes_id_form()) == {frozenset([1]), frozenset([2]), frozenset([3])}
    assert set(graph_rep_1.get_edges_id_form()) == {(frozenset([1]), frozenset([2])), (frozenset([2]), frozenset([3]))}
    graph_rep_1.add_node(4)
    graph_rep_1.add_edge(3, 4)
    assert set(graph_rep_1.get_edges_id_form()) == {(frozenset([1]), frozenset([2])), (frozenset([2]), frozenset([3])),
                                                  (frozenset([3]), frozenset([4]))}


@visualization
def test_graph_draw(graph_rep_1):
    graph_rep_1.add_node(4)
    graph_rep_1.add_edge(3, 4)
    graph_rep_1.draw()


def test_get_node_by_id_no_id(graph_rep_1):
    assert isinstance(graph_rep_1.get_node_by_id(1), QuNode)
    assert isinstance(graph_rep_1.get_node_by_id(frozenset([1])), QuNode)

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
