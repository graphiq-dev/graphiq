from graphiq.backends.graph.state import Graph
import networkx as nx


def test_graph_initialization():
    graph = nx.Graph([(0, 1), (1, 2), (1, 3), (2, 3)])
    state = Graph(graph)
    state.draw()
    assert state.n_qubits == 4
    print(f"Neighbors of node 1 are {state.get_neighbors(1)} ")
    lc_state = state.local_complementation(1, copy=True)
    state.draw()
    lc_state.draw()


def test_graph_lc():
    lc_dict = {0: ["x"], 1: ["h"], 2: ["s"], 3: ["s", "h"]}
    graph = nx.Graph([(0, 1), (1, 2), (1, 3), (2, 3)])
    state = Graph(graph, lc_dict)
    print(f"Local Clifford gate at node 1 is {state.find_lc(1)}")
