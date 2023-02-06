from src.utils.relabel_module import *
import networkx as nx
import numpy as np
from benchmarks.graph_states import (
    repeater_graph_states,
    star_graph_state,
    lattice_cluster_state,
)
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from math import ceil
from src.backends.stabilizer.functions.height import height_max

from src.backends.stabilizer.compiler import StabilizerCompiler
import src.backends.state_representation_conversion as converter
from src.backends.stabilizer.tableau import CliffordTableau
from src.metrics import Infidelity
from src.state import QuantumState
from src.solvers.deterministic_solver import DeterministicSolver


# %%
# TODO: rnd graph generator class to make many graphs of certain types or classes or different random types
def rnd_graph(n, p=None, m=None, seed=None, model="erdos"):
    # generates random connected graphs
    if model == "erdos" or model is None:
        rnd_maker = nx.gnp_random_graph
        x = p
    elif model == "albert":
        rnd_maker = nx.barabasi_albert_graph
        x = m
    g = nx.Graph()
    g.add_edges_from(((0, 1), (2, 3)))
    while not nx.is_connected(g):
        g = rnd_maker(n, x, seed=seed)
    return g


def graph_to_circ(graph, show=False):
    if not isinstance(graph, nx.Graph):
        graph = nx.from_numpy_array(graph)
        assert isinstance(graph, nx.Graph), "input must be a networkx graph object or a numpy adjacency matrix"
    n = graph.number_of_nodes()
    c_tableau = CliffordTableau(n)
    c_tableau.stabilizer_from_labels(converter.graph_to_stabilizer(graph))
    ideal_state = QuantumState(n, c_tableau, representation="stabilizer")

    compiler = StabilizerCompiler()
    target = ideal_state
    metric = Infidelity(target)
    solver = DeterministicSolver(
        target=target,
        metric=metric,
        compiler=compiler,
    )
    solver.solve()
    score, circ = solver.result
    if show:
        fig, (ax1, ax2) = plt.subplots(2, 1, dpi=300)
        nx.draw_networkx(
            graph, with_labels=True, pos=nx.kamada_kawai_layout(graph), ax=ax1
        )
        circ.draw_circuit(ax=ax2)
    return circ


def graph_to_depth(graph, emit_depth=False, show=False):
    # returns the longest path in DAG
    circ = graph_to_circ(graph)
    return circ.depth


def graph_to_emit_depth(graph):
    # returns the maximum depth of an emitter qubit between all emitter qubits
    assert isinstance(graph, nx.Graph)
    circ = graph_to_circ(graph)
    depth = {}
    for e_i in range(circ.n_emitters):
        depth[e_i] = len(circ.reg_gate_history(reg=e_i)[1])
    return max(depth.values())


def graph_to_cnot(graph):
    # returns the number of emitter-emitter cnot gates in the circuit
    circ = graph_to_circ(graph)
    cnot_nodes = circ.get_node_by_labels(["Emitter-Emitter", "CNOT"])
    return len(cnot_nodes)


# %% initialize
m = 16
# graph = lattice_cluster_state((3, 4)).data
# graph = star_graph_state(m).data
# graph = repeater_graph_states(m)
# graph = nx.random_tree(m)
graph = rnd_graph(m, 0.95, seed=None)
g1 = (nx.to_numpy_array(graph)).astype(int)
g2 = nx.from_numpy_array(g1)
g3 = nx.convert_node_labels_to_integers(g2)
adj = nx.to_numpy_array(g3)
n = len(g3)


# %% Utils
def adj_emit_list(adj, count=1000):
    # makes a number of isomorphic graphs out of the input one and returns both the adj list and #emitters list both
    # are sorted and there is one to one correspondence between the two lists.
    n = adj.shape[0]
    iso_count = min(np.math.factorial(n), count)
    arr = iso_finder(adj, iso_count, allow_exhaustive=True)
    li11 = emitter_sorted(arr)
    emit_list = [x[1] for x in li11]
    adj_list = [x[0] for x in li11]
    return adj_list, emit_list


def avg_maker(x_data, y_data):
    # for all similar items in data1 find the average of corresponding values in data2.
    data_dict = (
        {}
    )  # key is item in data1 and value is tuple (avg_so_far, count_of_items) from data2
    x_data_uniq = []
    y_data_avg = []
    y_data_std = []
    assert len(x_data) == len(y_data)
    for i in range(len(x_data)):
        x_value = x_data[i]
        if x_value in data_dict:
            # update
            data_dict[x_value].append(y_data[i])

        else:
            data_dict[x_value] = [y_data[i]]

    for item in sorted(data_dict.items()):
        x_data_uniq.append(item[0])
        y_data_avg.append(np.mean(item[1]))
        y_data_std.append(np.std(item[1]))
    return x_data_uniq, y_data_avg, y_data_std


# %% node based
# this section is used to see correlations of node based metrics with emission orders.


class Node_corr:
    def __init__(self, graph, relabel_trials=None, metric=None):
        if isinstance(graph, nx.Graph):
            self._graph = graph
            self._adj = nx.to_numpy_array(graph)
        elif isinstance(graph, np.ndarray):
            self._graph = nx.from_numpy_array(graph)
            self._adj = graph
        else:
            raise ValueError(
                "The input graph must be a networkx Graph or a valid adjacency matrix"
            )
        if relabel_trials is None:
            self._relabel_trials = 1000
        elif isinstance(relabel_trials, int):
            self._relabel_trials = relabel_trials
        else:
            raise ValueError("The input trial should be an integer")
        if metric is None:
            self._metric = "deg"
        elif isinstance(metric, str):
            self._metric = metric
        else:
            raise ValueError("The metric should be a valid string")

    @property
    def graph(self):
        return self._graph

    @property
    def adj(self):
        return self._adj

    @property
    def relabel_trials(self):
        return self._relabel_trials

    @relabel_trials.setter
    def relabel_trials(self, trials):
        assert isinstance(trials, int)
        self._relabel_trials = trials

    @property
    def metric(self):
        return self._metric

    @metric.setter
    def metric(self, new_met):
        assert isinstance(new_met, str)
        self._metric = new_met

    def order_corr(self):
        adj_list, emit_list = adj_emit_list(self.adj, count=self.relabel_trials)
        data1 = [*range(self.adj.shape[0])]
        pear_corr_list = []
        for new_adj in adj_list:
            new_g = nx.from_numpy_array(new_adj)
            # dict_centrality = nx.betweenness_centrality(new_g)
            dict_centrality = self.metric_node_dict(new_g)
            data2 = [dict_centrality[x] for x in data1]
            # calculate Pearson's correlation
            corr, _ = pearsonr(data1, data2)
            pear_corr_list.append(abs(corr))
        overall_corr, _ = pearsonr(emit_list, pear_corr_list)
        print("correlation:", overall_corr)
        # retry with average values over all cases with the same number of emitters
        emits_no_repeat, avg_corrs, std_corrs = avg_maker(emit_list, pear_corr_list)
        overall_avg_corr, _ = pearsonr(emits_no_repeat, avg_corrs)
        print("avg correlation:", overall_avg_corr)
        plt.scatter(emits_no_repeat, avg_corrs)
        plt.errorbar(
            emits_no_repeat,
            avg_corrs,
            yerr=std_corrs,
            fmt="bo",
            ecolor="r",
            capsize=4,
            errorevery=None,
            capthick=2,
        )
        plt.show()
        return emits_no_repeat, avg_corrs, std_corrs

    def metric_node_dict(self, graph):
        """
        Calculates the dictionary of nodes and their corresponding node-based graph metric values, e.g. degree of each
        node.
        :return: dictionary of nodes and metric values of each node
        :rtype: dict
        """
        metric = self.metric
        if metric == "deg":
            return dict(graph.degree)
        elif metric == "bet":
            return nx.betweenness_centrality(graph)
        elif metric == "close":
            return nx.closeness_centrality(graph)
        elif metric == "eigen":
            return nx.eigenvector_centrality(graph)
        elif metric == "nei_deg":
            return nx.average_neighbor_degree(graph)
        elif metric == "eccentric":
            return nx.eccentricity(graph)
        elif metric == "cluster":
            return nx.clustering(graph)
        elif metric == "pagerank":
            return nx.pagerank(graph)
        else:
            raise NotImplementedError(
                f"Input metric {self.metric} not found. It may not be implemented"
            )


def between_corr(adj, count=10000):
    adj_list, emit_list = adj_emit_list(adj, count=count)
    data1 = [*range(n)]
    pear_corr_list = []
    for new_adj in adj_list:
        new_g = nx.from_numpy_array(new_adj)
        dict_centrality = nx.betweenness_centrality(new_g)
        # dict_centrality = nx.load_centrality(new_g)
        data2 = [dict_centrality[x] for x in data1]
        # calculate Pearson's correlation
        corr, _ = pearsonr(data1, data2)
        pear_corr_list.append(abs(corr))
    overall_corr, _ = pearsonr(emit_list, pear_corr_list)
    print("correlation:", overall_corr)
    # retry with average values over all cases with the same number of emitters
    emits_no_repeat, avg_corrs, std_corrs = avg_maker(emit_list, pear_corr_list)
    overall_avg_corr, _ = pearsonr(emits_no_repeat, avg_corrs)
    print("avg correlation:", overall_avg_corr)
    plt.scatter(emits_no_repeat, avg_corrs)
    plt.errorbar(
        emits_no_repeat,
        avg_corrs,
        yerr=std_corrs,
        fmt="bo",
        ecolor="r",
        capsize=4,
        errorevery=4,
        capthick=2,
    )
    plt.show()
    return emits_no_repeat, avg_corrs


def nei_deg_corr(adj, count=1000):
    # metric used: mean neighbors degree.
    return


# %%
def num_emit_vs_p(n, trials=1000, p_step=0.1):
    # Average num of emitters vs “p” for random graphs of a certain size “n”
    num_emit_list = []
    avg_emit = []
    avg_emit_std = []
    p_list = [x * p_step for x in range(ceil(0.1 / p_step), int(1 / p_step))]
    for p in p_list:
        for i in range(trials):
            g = rnd_graph(n, p)
            n_emit = height_max(graph=g)
            num_emit_list.append(n_emit)

        avg_emit.append(np.mean(num_emit_list))
        avg_emit_std.append(np.std(num_emit_list))
        num_emit_list = []
    fig = plt.figure(figsize=(7.5, 5.5), dpi=300)
    fig.tight_layout()
    plt.scatter(p_list, avg_emit)
    plt.errorbar(
        p_list,
        avg_emit,
        yerr=list(avg_emit_std),
        fmt="bo",
        ecolor="r",
        capsize=4,
        errorevery=4,
        capthick=2,
    )
    plt.title("Number of emitters vs Edge probability for a graph")
    plt.figtext(0.77, 0.79, f"n: {n}\ntrials:{'{:.0e}'.format(trials)}")
    plt.xlabel("Edge probability")
    plt.ylabel("Number of emitters")
    plt.ylim(0, max(avg_emit) * 1.25)
    plt.show()
    return p_list, avg_emit, avg_emit_std


def num_emit_dist(n, p, trials=1000, show_plot=False):
    num_emit_list = []
    for i in range(trials):
        g = rnd_graph(n, p)
        n_emit = height_max(graph=g)
        num_emit_list.append(n_emit)
    num_emit_set = set(num_emit_list)
    count_percent_list = [(num_emit_list.count(x) / trials) * 100 for x in num_emit_set]
    dist_dict = dict(zip(num_emit_set, count_percent_list))  # {num_emit: its count}
    avg = np.mean(num_emit_list)
    std = np.std(num_emit_list)
    print("mean and standard deviation:", avg, ",", std)
    if show_plot:
        fig = plt.figure(figsize=(8, 6), dpi=300)
        fig.tight_layout()
        plt.scatter(dist_dict.keys(), dist_dict.values())
        plt.figtext(
            0.75,
            0.70,
            f"n: {n}\np: {p}\navg: {round(avg, 2)}\nstd: {round(std, 2)}\ntrials: "
            f"{'{:.0e}'.format(trials)}",
        )
        plt.title("Number of emitters for random graphs")
        plt.xlabel("Number of emitters")
        plt.ylabel("% percentage")
        plt.xticks(range(min(num_emit_set), max(num_emit_set) + 1))
        plt.ylim(0, max(count_percent_list) * 1.25)
        plt.show()
    return avg, std


def num_emit_vs_n(n_range, n_p, trials=100):  # for constant n*p in the erdos models
    # n_range is a tuple (min num of nodes, max num of nodes) to cover
    num_emit_list = []
    avg_emit = []
    avg_emit_std = []
    assert n_p / n_range[0] <= 1
    n_list = [*range(n_range[0], n_range[1])]
    for n in n_list:
        for i in range(trials):
            g = rnd_graph(n, n_p / n)
            n_emit = height_max(graph=g)
            num_emit_list.append(n_emit)
        avg_emit.append(np.mean(num_emit_list))
        avg_emit_std.append(np.std(num_emit_list))
        num_emit_list = []
    fig = plt.figure(figsize=(7.5, 5.5), dpi=300)
    fig.tight_layout()
    plt.scatter(n_list, avg_emit)
    plt.errorbar(
        n_list,
        avg_emit,
        yerr=list(avg_emit_std),
        fmt="bo",
        ecolor="r",
        capsize=4,
        errorevery=4,
        capthick=2,
    )
    plt.title("Number of emitters vs Size\nfor constant degree")
    plt.figtext(0.76, 0.79, f"n*p: {n_p}\ntrials:{'{:.0e}'.format(trials)}")
    plt.xlabel("Number of nodes")
    plt.ylabel("Number of emitters")
    plt.ylim(0, max(avg_emit) * 1.25)
    plt.show()
    return n_list, avg_emit, avg_emit_std


# %% circ depth vs n and p and constant np


def depth_vs_p(n, trials=1000, p_step=0.1):
    # Average num of emitters vs “p” for random graphs of a certain size “n”
    depth_list = []
    avg_depth = []
    avg_depth_std = []
    p_list = [x * p_step for x in range(ceil(0.1 / p_step), int(1 / p_step))]
    for p in p_list:
        for i in range(trials):
            g = rnd_graph(n, p)
            depth = graph_to_depth(graph=g)
            depth_list.append(depth)

        avg_depth.append(np.mean(depth_list))
        avg_depth_std.append(np.std(depth_list))
        depth_list = []
    fig = plt.figure(figsize=(7.5, 5.5), dpi=300)
    fig.tight_layout()
    plt.scatter(p_list, avg_depth)
    plt.errorbar(
        p_list,
        avg_depth,
        yerr=list(avg_depth_std),
        fmt="bo",
        ecolor="r",
        capsize=4,
        errorevery=4,
        capthick=2,
    )
    plt.title("Depth vs Edge probability for a graph")
    plt.figtext(0.77, 0.79, f"n: {n}\ntrials:{'{:.0e}'.format(trials)}")
    plt.xlabel("Edge probability")
    plt.ylabel("Circuit depth")
    plt.ylim(0, max(avg_depth) * 1.25)
    plt.show()
    return p_list, avg_depth, avg_depth_std


def depth_dist(n, p, trials=1000, show_plot=False):
    depth_list = []
    for i in range(trials):
        g = rnd_graph(n, p)
        depth = graph_to_depth(graph=g)
        depth_list.append(depth)
    num_emit_set = set(depth_list)
    count_percent_list = [(depth_list.count(x) / trials) * 100 for x in num_emit_set]
    dist_dict = dict(zip(num_emit_set, count_percent_list))  # {num_emit: its count}
    avg = np.mean(depth_list)
    std = np.std(depth_list)
    print("mean and standard deviation:", avg, ",", std)
    if show_plot:
        fig = plt.figure(figsize=(8, 6), dpi=300)
        fig.tight_layout()
        plt.scatter(dist_dict.keys(), dist_dict.values())
        plt.figtext(
            0.75,
            0.70,
            f"n: {n}\np: {p}\navg: {round(avg, 2)}\nstd: {round(std, 2)}\ntrials: "
            f"{'{:.0e}'.format(trials)}",
        )
        plt.title("Depth for random graphs")
        plt.xlabel("Circuit depth")
        plt.ylabel("% percentage")
        plt.xticks(range(min(num_emit_set), max(num_emit_set) + 1))
        plt.ylim(0, max(count_percent_list) * 1.25)
        plt.show()
    return avg, std


# %% graph based: max betweenness


# %%
class Graph_corr:
    def __init__(
            self, graph_metric=None, circ_metric=None, initial_graph=None, graph_list=None, trials=None,
            relabel_trials=None, num_isomorph=1
    ):
        # if we have an initial graph, correlations are found based on the relabeling of that graph. Otherwise, if a
        # graph_list is given, metrics are analyzed for those graphs. If neither are present, correlations are found
        # for random graphs based on n and p provided in the finder method input arguments.
        self._num_isomorph = num_isomorph
        if trials is None:
            self._trials = 1000
        elif isinstance(trials, int):
            self._trials = trials
        else:
            raise ValueError("The input trial should be an integer")
        if relabel_trials is None:
            self._relabel_trials = None
        elif isinstance(relabel_trials, int):
            self._relabel_trials = relabel_trials
        else:
            raise ValueError("The input relabel trial should be an integer")
        if graph_metric is None:
            self._graph_metric = "deg"
        elif isinstance(graph_metric, str):
            self._graph_metric = graph_metric
        else:
            raise ValueError("The graph metric should be a valid string")
        if circ_metric is None:
            self._circ_metric = "num_emit"
        elif isinstance(circ_metric, str):
            self._circ_metric = circ_metric
        else:
            raise ValueError("The circuit metric should be a valid string")

        if initial_graph is None:
            self.graph_list = graph_list
            assert isinstance(graph_list[0], nx.Graph), "graph_list must be a list of networkx graphs"
        else:
            self._initial_graph = initial_graph
            assert isinstance(initial_graph, nx.Graph), "initial graph must be a networkx graph object"
            self.graph_list = self._graph_list_maker(self._initial_graph, count=self._num_isomorph)

    @property
    def initial_graph(self):
        return self._initial_graph

    @initial_graph.setter
    def initial_graph(self, g):
        assert isinstance(g, nx.Graph), "initial graph must be a networkx graph object"
        self._initial_graph = g
        self.graph_list = self._graph_list_maker(self._initial_graph, count=self._num_isomorph)

    @property
    def num_isomorph(self):
        return self._num_isomorph

    @num_isomorph.setter
    def num_isomoprh(self, num):
        assert isinstance(num, int)
        self._num_isomorph = num
        self.graph_list = self._graph_list_maker(self._initial_graph, count=num)

    @property
    def relabel_trials(self):
        return self._relabel_trials

    @relabel_trials.setter
    def relabel_trials(self, trials):
        assert isinstance(trials, int) or trials is None
        self._relabel_trials = trials

    @property
    def trials(self):
        return self._trials

    @trials.setter
    def trials(self, new):
        assert isinstance(new, int)
        self._trials = new

    @property
    def graph_metric(self):
        return self._graph_metric

    @graph_metric.setter
    def graph_metric(self, new_met):
        assert isinstance(new_met, str)
        self._graph_metric = new_met

    @property
    def circ_metric(self):
        return self._circ_metric

    @circ_metric.setter
    def circ_metric(self, new_met):
        assert isinstance(new_met, str)
        self._circ_metric = new_met

    def finder(self, graph_type=None, show_plot=False, **kwargs):
        """check correlation between the graph and circuit metric values for a set of graphs"""
        graph_met_list = []
        circ_met_list = []
        if self.graph_list is None:
            for i in range(self.trials):
                g = rnd_graph(kwargs['n'], kwargs['p'], model=graph_type)
                graph_value = self._graph_met_value(g)
                circ_value = self._min_circ_met_over_relabel(g)
                graph_met_list.append(graph_value)
                circ_met_list.append(circ_value)
        else:
            for g in self.graph_list:
                graph_value = self._graph_met_value(g)
                circ_value = self._min_circ_met_over_relabel(g)
                graph_met_list.append(graph_value)
                circ_met_list.append(circ_value)
        corr, _ = pearsonr(circ_met_list, graph_met_list)
        # average case
        circ_met_uniq, avg_graph_met, std_graph_met = avg_maker(circ_met_list, graph_met_list)
        avg_corr, _ = pearsonr(circ_met_uniq, avg_graph_met)
        if show_plot:
            fig, (ax1, ax2) = plt.subplots(2, figsize=(15, 15))
            ax1.scatter(circ_met_list, graph_met_list)
            ax2.plot(circ_met_uniq, avg_graph_met)
            ax2.set_xlabel(f"{self.circ_metric}")
            ax2.set_ylabel(f"average {self.graph_metric}")
            ax2.errorbar(
                circ_met_uniq,
                avg_graph_met,
                yerr=std_graph_met,
                fmt="bo",
                ecolor="r",
                capsize=4,
                errorevery=1,
                capthick=2,
            )
            plt.show()
            print("correlation=", corr, "avg correlation=", avg_corr)
        return corr, avg_corr, (circ_met_uniq, avg_graph_met)

    def corr_p_dependence(self, n, p_step=0.1):
        p_list = [x * p_step for x in range(ceil(0.1 / p_step), int(1 / p_step))]
        for p in p_list:
            _, _, plot_pair = self.finder(n=n, p=p)
            plt.plot(plot_pair[0], plot_pair[1], label=f"p={round(p, 2)}")
        plt.title(f"the {self.graph_metric} - {self.circ_metric} correlation for different p values")
        plt.legend(loc="best")
        plt.yscale("log")
        plt.show()

    def corr_n_dependence(self, p, n_step=10):
        # number of nodes starting from 5 and going up to at least 100, or possibly 10 times the chosen n_step
        n_list = [*range(5, max(10 * n_step, 100) + 1, n_step)]
        for n in n_list:
            _, _, plot_pair = self.finder(n=n, p=p)
            plt.plot(plot_pair[0], plot_pair[1], label=f"n={n}")
        plt.title(f"the {self.graph_metric} - {self.circ_metric} correlation for different n values")
        plt.legend(loc="best")
        plt.yscale("log")
        plt.show()

    def met_distribution(self, n=10, p=0.5, which_met="circ", show_plot=False):
        met_list = []
        if self.graph_list is None:
            for i in range(self.trials):
                g = rnd_graph(n, p)
                if which_met == "graph":
                    met_value = self._graph_met_value(g)
                elif which_met == "circ":
                    met_value = self._min_circ_met_over_relabel(g)
                else:
                    raise ValueError("Choose between 'graph' or 'circ' options for which_met")
                met_list.append(met_value)
        else:
            for g in self.graph_list:
                if which_met == "graph":
                    met_value = self._graph_met_value(g)
                elif which_met == "circ":
                    met_value = self._min_circ_met_over_relabel(g)
                else:
                    raise ValueError("Choose between 'graph' or 'circ' options for which_met")
                met_list.append(met_value)
        met_set = set(met_list)
        count_percent_list = [(met_list.count(x) / self.trials) * 100 for x in met_set]
        dist_dict = dict(zip(met_set, count_percent_list))  # {num_emit: its count}
        avg = np.mean(met_list)
        std = np.std(met_list)
        print("mean and standard deviation:", avg, ",", std)
        if show_plot:
            fig = plt.figure(figsize=(8, 6), dpi=300)
            fig.tight_layout()
            plt.scatter(dist_dict.keys(), dist_dict.values())
            plt.figtext(
                0.75,
                0.70,
                f"n: {n}\np: {p}\navg: {round(avg, 2)}\nstd: {round(std, 2)}\ntrials: "
                f"{'{:.0e}'.format(self.trials)}",
            )
            if which_met == "circ":
                plt.title(f"{self.circ_metric} for random graphs")
                plt.xlabel(f"{self.circ_metric}")
            else:
                plt.title(f"{self.graph_metric} for random graphs")
                plt.xlabel(f"{self.graph_metric}")
            plt.ylabel("% percentage")
            plt.xticks(range(min(met_set), max(met_set) + 1))
            plt.ylim(0, max(count_percent_list) * 1.25)
            plt.show()
        return avg, std

    @staticmethod
    def _graph_list_maker(g, count):
        # makes a list of relabeled graphs out of the initial one. List size = count
        adj = nx.to_numpy_array(g)
        nod = g.number_of_nodes()
        iso_count = min(np.math.factorial(nod), count)
        adj_arr = iso_finder(adj, iso_count, allow_exhaustive=True)
        graph_list = [g]
        for adj_i in adj_arr:
            g_i = nx.from_numpy_array(adj_i)
            graph_list.append(g_i)
        return graph_list

    def _min_circ_met_over_relabel(self, g):
        """Find the circuit metric over a number of isomorphic graphs and return the minimum value found."""
        if self.relabel_trials:
            circ_value_list = []
            g_list = self._graph_list_maker(g, count=self.num_isomorph)
            for g_i in g_list:
                circ_value_list.append(self._circ_met_value(g_i))
            circ_value = min(circ_value_list)
            return circ_value
        else:
            return self._circ_met_value(g)

    def _graph_met_value(self, g):
        if self.graph_metric == "max_between":
            dict_centrality = nx.betweenness_centrality(g)
            graph_value = max(dict_centrality.values())
        elif self.graph_metric == "mean_between":
            dict_centrality = nx.betweenness_centrality(g)
            graph_value = np.mean(list(dict_centrality.values()))
        elif self.graph_metric == "mean_nei_deg":
            # the mean of the "average neighbors degree" over all nodes in graph
            dict_met = nx.average_neighbor_degree(g)
            graph_value = np.mean(list(dict_met.values()))
        elif self.graph_metric == "mean_deg":
            dict_met = dict(g.nodes())
            graph_value = np.mean(list(dict_met.values()))
        elif self.graph_metric == "node_connect":
            graph_value = nx.node_connectivity(g)
        elif self.graph_metric == "edge_connect":
            graph_value = nx.edge_connectivity(g)
        elif self.graph_metric == "assort":
            graph_value = nx.degree_assortativity_coefficient(g)
        elif self.graph_metric == "radius":
            graph_value = nx.radius(g)
        elif self.graph_metric == "diameter":
            graph_value = nx.diameter(g)
        elif self.graph_metric == "periphery":
            # num of nodes with distance equal to diameter
            graph_value = len(nx.periphery(g))
        elif self.graph_metric == "center":
            # num of nodes with distance equal to radius
            graph_value = len(nx.center(g))
        elif self.graph_metric == "cluster":
            graph_value = nx.average_clustering(g)
        elif self.graph_metric == "local_efficiency":
            graph_value = nx.local_efficiency(g)
        elif self.graph_metric == "global_efficiency":
            graph_value = nx.global_efficiency(g)
        elif self.graph_metric == "node":
            graph_value = g.number_of_nodes()
        elif self.graph_metric == "pop":
            nodes = g.number_of_nodes()
            edges = g.size()
            graph_value = edges / ((nodes * (nodes - 1)) / 2)
        else:
            raise ValueError(
                f"Graph metric {self.graph_metric} not found. It may not be implemented"
            )

        return graph_value

    def met_met_plot(self, met1, met2):
        #TODO

    def _circ_met_value(self, g):
        if self.circ_metric == "num_emit":
            circ_value = height_max(graph=g)
        elif self.circ_metric == "depth":
            circ_value = graph_to_depth(graph=g)
        elif self.circ_metric == "num_emit_per_photon":
            circ_value = height_max(graph=g) / g.number_of_nodes()
        elif self.circ_metric == "max_emit_depth":
            circ_value = graph_to_emit_depth(graph=g)
        elif self.circ_metric == "cnot":
            circ_value = graph_to_cnot(graph=g)
        elif self.circ_metric == "cnot_per_photon":
            circ_value = graph_to_cnot(graph=g) / len([g])
        elif self.circ_metric == "cnot_per_emitter":
            circ_value = graph_to_cnot(graph=g) / height_max(graph=g)
        else:
            raise ValueError(
                f"Circuit metric {self.circ_metric} not found. It may not be implemented"
            )
        return circ_value


# %%
# between_corr(adj, count=1000)

# %% whole graph based
# p_of_edge_dependence(24, n_samples=500, relabel_trials=4)
# %%
# max_bet_min_emit_corr(n, 0.9, n_samples=100, relabel_trials=100)
# num_emit_vs_n((10, 40), 9, trials=80)
# graph_to_depth(rnd_graph(5, 0.3))

# %% visual
# fig, (ax1, ax2) = plt.subplots(2, figsize=(15, 25))
# fig.tight_layout()
# nx.draw_networkx(g3, with_labels=1, ax=ax1, pos=nx.kamada_kawai_layout(g3))
# nx.draw_networkx(new_g, with_labels=1, ax=ax2, pos=nx.kamada_kawai_layout(new_g))
# plt.show()

# %%
import time

graph = rnd_graph(16, 0.08)
# get the start time

circ = graph_to_circ(graph)

st = time.time()
dd = circ.calculate_reg_depth("e")
et = time.time()
d = {}
for i in range(circ.n_emitters):
    d[i] = len(circ.reg_gate_history(reg=i)[1])
et2 = time.time()
elapsed_time1 = et - st
elapsed_time2 = et2 - st
print(d, "\n", dd,
      "Execution time without and with:",
      elapsed_time1,
      elapsed_time2,
      elapsed_time2 / elapsed_time1,
      "seconds",
      )
