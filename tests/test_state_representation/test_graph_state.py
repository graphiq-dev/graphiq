from src.backends.graph.state import Graph
from src.visualizers.graph import *
from tests.test_flags import visualization


def test_graph_initialization():
    g = nx.Graph([(1, 2), (2, 3)])
    graph_rep = Graph(data=g, root_node_id=1)

    assert set(graph_rep.get_nodes_id_form()) == {frozenset([1]), frozenset([2]), frozenset([3])}
    assert set(graph_rep.get_edges_id_form()) == {(frozenset([1]), frozenset([2])), (frozenset([2]), frozenset([3]))}
    graph_rep.add_node(4)
    graph_rep.add_edge(3, 4)
    assert set(graph_rep.get_edges_id_form()) == {(frozenset([1]), frozenset([2])), (frozenset([2]), frozenset([3])),
                                                  (frozenset([3]), frozenset([4]))}


@visualization
def test_graph_draw():
    g = nx.Graph([(1, 2), (2, 3)])
    graph_rep = Graph(data=g, root_node_id=1)
    graph_rep.add_node(4)
    graph_rep.add_edge(3, 4)
    graph_rep.draw()
