import matplotlib.pyplot as plt
from graphiq.benchmarks.alternate_circuits import exemplary_run, report_alternate_circuits
from graphiq.benchmarks.graph_states import repeater_graph_states
from graphiq.utils.preprocessing import *


def run_multiple_graphs_and_report(graph, lc_nodes):
    results = [exemplary_run(graph, None)]
    graphs = [graph]
    for node in lc_nodes:
        alt_graph = local_comp_graph(graph, node)

        graphs.append(alt_graph)
        results.append(exemplary_run(graph, None))
    for i in range(len(results)):
        ax = plt.axes()
        nx.draw(graphs[i], ax=ax, with_labels=True)
        report_alternate_circuits(results[i])


def linear_4_example():
    graph = nx.Graph([(0, 1), (1, 2), (2, 3)])
    run_multiple_graphs_and_report(graph, [1, 2])


def square_4_example():
    graph = nx.Graph([(0, 1), (0, 2), (2, 3), (1, 3), (1, 2)])
    run_multiple_graphs_and_report(graph, [0, 1, 2, 3])


def repeater_graph_example(n_inner_qubits):
    graph = repeater_graph_states(n_inner_qubits)
    run_multiple_graphs_and_report(graph, [1])


def star_graph_example(n_points):
    graph = nx.star_graph(n_points)
    run_multiple_graphs_and_report(graph, [0, 1])


def biclique_example(n_pairs):
    edges = []
    for i in range(n_pairs):
        for j in range(n_pairs):
            edges.append((i, j + n_pairs))
    graph = nx.Graph(edges)
    run_multiple_graphs_and_report(graph, [0, 1, n_pairs])


def imperfect_repeater_graph_example():
    edges_set1 = [(2, 1), (2, 4), (2, 6), (2, 8), (2, 10)]
    edges_set1 += [(0, 1), (3, 4), (5, 6), (7, 8), (9, 10)]
    edges_set2 = edges_set1.copy()
    node_set = [1, 4, 6, 8, 10]
    for i in range(len(node_set)):
        for j in range(i + 1, len(node_set)):
            edges_set1.append((node_set[i], node_set[j]))
    graph1 = nx.Graph(edges_set1)
    nx.draw(graph1, with_labels=True)
    result = exemplary_run(graph1, None)
    report_alternate_circuits(result)
    graph2 = nx.Graph(edges_set2)
    nx.draw(graph2, with_labels=True)
    result = exemplary_run(graph2, None)
    report_alternate_circuits(result)


def heuristic_example1():
    graph = nx.complete_graph(5)
    nx.draw(graph, with_labels=True)
    result = exemplary_run(graph, None)
    report_alternate_circuits(result)

    graph2 = get_lc_graph_by_max_edge(graph, 10)
    nx.draw(graph2, with_labels=True)
    result = exemplary_run(graph2, None)
    report_alternate_circuits(result)


def heuristic_example2():
    graph = nx.complete_bipartite_graph(3, 3)
    nx.draw(graph, with_labels=True)
    result = exemplary_run(graph, None)
    report_alternate_circuits(result)

    graph2 = get_lc_graph_by_max_edge(graph, 3)
    # nx.draw(graph2, with_labels=True)
    # result = exemplary_run(graph2, None)
    # report_alternate_circuits(result)
    graph3 = local_comp_graph(graph2, 0)
    nx.draw(graph3, with_labels=True)
    result = exemplary_run(graph3, None)
    report_alternate_circuits(result)


def heuristic_example3():
    graph = nx.complete_bipartite_graph(3, 3)
    nx.draw(graph, with_labels=True)
    result = exemplary_run(graph, None)
    report_alternate_circuits(result)

    # graph2 = candidate_graphs(graph, 3)
    graph2 = get_lc_graph_by_max_neighbor_edge(graph, 3)
    nx.draw(graph2, with_labels=True)
    result = exemplary_run(graph2, None)
    report_alternate_circuits(result)


if __name__ == "__main__":
    # linear_4_example()
    # square_4_example()
    # repeater_graph_example(4)
    # star_graph_example(7)
    # biclique_example(3)
    # imperfect_repeater_graph_example()
    heuristic_example3()
