import pytest
from src.utils.correlation_module import *


@pytest.mark.parametrize(
    "metric",
    ["deg", "bet", "close", "eigen", "nei_deg", "eccentric", "cluster", "pagerank"],
)
@pytest.mark.parametrize("circ_met", ["num_emit", "cnot", "depth"])
def test_node_corr(metric, circ_met):
    """test different metrics and change of graph and number of relabeling trials"""
    correlation = NodeCorr(
        rnd_graph(n=4, p=0.7),
        relabel_trials=2,
        metric=metric,
        circ_met=circ_met,
        show_plot=False,
    )
    correlation.order_corr()
    correlation.graph = rnd_graph(n=4, p=0.2)
    correlation.met_order_error()
    correlation.relabel_trials = 1
    if metric != "eccentric":
        correlation.next_node_corr()


@pytest.mark.parametrize(
    "graph_metric",
    [
        "max_between",
        "mean_nei_deg",
        "mean_deg",
        "node_connect",
        "edge_connect",
        "assort",
        "radius",
        "center",
        "periphery",
        "cluster",
        "local_efficiency",
    ],
)
@pytest.mark.parametrize("circ_metric", ["num_emit", "cnot", "depth"])
def test_random_graph_corr(graph_metric, circ_metric):
    correlation = GraphCorr(
        graph_metric=graph_metric, circ_metric=circ_metric, trials=4
    )
    correlation.finder(n=4, p=0.6, show_plot=False)
    correlation.met_distribution(n=4, p=0.6, show_plot=False)


def test_initial_graph_corr():
    g = rnd_graph(n=4, p=0.6)
    correlation = GraphCorr(
        graph_metric="mean_nei_deg",
        circ_metric="depth",
        initial_graph=g,
        num_isomorph=2,
    )
    correlation.finder(show_plot=False)
    correlation.met_met(met1="num_emit", met2="node_connect", show_plot=False)


def test_graph_list_corr():
    graph_list = [rnd_graph(n=4, p=i) for i in [0.2, 0.4, 0.6, 0.8]]
    correlation = GraphCorr(
        graph_metric="mean_nei_deg", circ_metric="num_emit", graph_list=graph_list
    )
    correlation.finder(show_plot=False)
    correlation.met_met(met1="num_emit", met2="node_connect", show_plot=False)
