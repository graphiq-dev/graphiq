from src.utils.preprocessing import *
from benchmarks.graph_states import *
from benchmarks.alternate_circuits import exemplary_run
from src.solvers.hybrid_solvers import HybridGraphSearchSolver
from src.backends.stabilizer.compiler import StabilizerCompiler


def run_candidate_graphs(original_graph, candidate_graphs):
    results = exemplary_run(original_graph, None)
    for score, graph in candidate_graphs:
        results += exemplary_run(graph, None)

    original_depths = results[0][2].calculate_reg_depth("e")
    print(f"The circuit emitter depths of the original graph is {original_depths}.")
    for i in range(1, len(results)):
        depths = results[i][2].calculate_reg_depth("e")
        print(f"The circuit emitter depths of the graph {i} is {depths}.")


def run_candidate_graphs_statistics(candidate_graphs):
    result_statistics = []
    for score, graph in candidate_graphs:
        results = exemplary_run(graph, None)
        for result in results:
            depths = result[2].calculate_reg_depth("e")
            result_statistics.append(max(depths))

    return result_statistics


def test_random_graph():
    n_qubit = 5
    n_test = 1
    edge_prob = 0.5
    for i in range(n_test):
        graph = random_graph_state(n_qubit, edge_prob)
        graph = graph.data

        candidate_graphs_1 = get_lc_graph_by_max_edge(graph, 3, graph_max_eccentricity)
        candidate_graphs_2 = get_lc_graph_by_max_neighbor_edge(
            graph, 3, graph_max_eccentricity
        )
        print("running the max edge approach")
        run_candidate_graphs(graph, candidate_graphs_1)
        print("running the max neighbor edge approach")
        run_candidate_graphs(graph, candidate_graphs_2)


def random_graph_run_setup(n_qubit, n_test, graph_metric):
    edge_prob = 0.5
    statistics_count1 = 0
    statistics_count2 = 0
    for i in range(n_test):
        graph = random_graph_state(n_qubit, edge_prob)
        graph = graph.data
        benchmark_statistics = graph_circuit_depth(graph)
        candidate_graphs_1 = get_lc_graph_by_max_edge(graph, 5, graph_metric)
        candidate_graphs_2 = get_lc_graph_by_max_neighbor_edge(graph, 5, graph_metric)

        statistics_1 = run_candidate_graphs_statistics(candidate_graphs_1)
        if min(statistics_1) < benchmark_statistics:
            statistics_count1 += 1

        statistics_2 = run_candidate_graphs_statistics(candidate_graphs_2)
        if min(statistics_2) < benchmark_statistics:
            statistics_count2 += 1
    print(f"Using the graph metric: {graph_metric}")
    print(f"The approach 1 has the statistics: {statistics_count1} out of {n_test}")
    print(f"The approach 2 has the statistics: {statistics_count2} out of {n_test}")


# def test_random_graph_statistics():
#    n_qubit = 5
#    n_test = 50
#    random_graph_run_setup(n_qubit, n_test, graph_max_eccentricity)
#    random_graph_run_setup(n_qubit, n_test, graph_max_centrality)
