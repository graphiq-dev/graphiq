from src.utils.relabel_module import *
import networkx as nx
import numpy as np
import random
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
    """
    A simple random connected graph generator. Two models are supported: Erdos-Renyi and Barabasi-Albert.
    :param n: Number of nodes in graph
    :type n: int
    :param p: The probability of edge existing between any two node. Used in Erdos-Renyi model.
    :type p: float
    :param m: The degree of each node added to the graph in the Barabasi-Albert model
    :type m: int
    :param seed: Seed for random generation
    :type seed: int
    :param model: Random graph model; either "erdos" or "albert"
    :type model: str
    :return: a random connected graph with specified properties
    :rtype: networkx.Graph
    """
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
    """
    Find a circuit that generates the input graph. This function calls the deterministic solver. The outcome is not
    unique.
    :param graph: The graph to generate
    :type graph: networkx.Graph
    :param show: If true draws the corresponding circuit
    :type show: bool
    :return: A circuit corresponding to the input graph
    :rtype: CircuitDAG
    """
    if not isinstance(graph, nx.Graph):
        graph = nx.from_numpy_array(graph)
        assert isinstance(
            graph, nx.Graph
        ), "input must be a networkx graph object or a numpy adjacency matrix"
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


def graph_to_depth(graph):
    """
    Finds the maximum depth of the whole circuit that generates the input graph. Note that it is different from the
    maximum number of gates applied on a single qubit in the circuit. The path can be shared between multiple qubits
    through multi-qubit gates.
    :param graph: The input graph
    :type graph: networkx.Graph
    :return: maximum depth
    :rtype: int
    """
    circ = graph_to_circ(graph)
    return circ.depth


def graph_to_emit_depth(graph):
    """
    Finds the maximum number of gates applied on a single emitter qubit in the circuit that generates the input graph.
    :param graph: The input graph
    :type graph: networkx.Graph
    :return: maximum depth of an emitter
    :rtype: int
    """
    # returns the maximum depth of an emitter qubit between all emitter qubits
    assert isinstance(graph, nx.Graph)
    circ = graph_to_circ(graph)
    depth = {}
    for e_i in range(circ.n_emitters):
        depth[e_i] = len(circ.reg_gate_history(reg=e_i)[1])
    return max(depth.values())


def graph_to_cnot(graph):
    """
    Finds the total number of emitter-emitter CNOT gates in the whole circuit that generates the input graph.
    :param graph: The input graph
    :type graph: networkx.Graph
    :return: number of CNOT gates between emitter qubits
    :rtype: int
    """
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
graph = rnd_graph(m, 0.75, seed=None)
g1 = (nx.to_numpy_array(graph)).astype(int)
g2 = nx.from_numpy_array(g1)
g3 = nx.convert_node_labels_to_integers(g2)
adj = nx.to_numpy_array(g3)
n = len(g3)
gg = repeater_graph_states(8)


# %% Utils
def adj_emit_sorted_list(adj, count=1000):
    """
    Finds as many as "count" number of isomorphic graphs out of the input one, if possible, and returns both a list of
    the adjacency matrices of those graph and a list containing the required number of emitters corresponding to each of
    the graphs in the same order. The graph in the list are sorted from the minimum to maximum number of emitters.
    :param adj: The adjacency matrix of the initial graph
    :type adj: numpy.ndarray
    :param count: The number of isomorph graphs one needs to make by relabeling the initial one.
    :type count: int
    :return: Two lists. The adjacency matrices list and the emitter list.
    :rtype: list, list
    """
    n = adj.shape[0]
    iso_count = min(np.math.factorial(n), count)
    arr = iso_finder(adj, iso_count, allow_exhaustive=True)
    li11 = emitter_sorted(arr)
    emit_list = [x[1] for x in li11]
    adj_list = [x[0] for x in li11]
    return adj_list, emit_list


def adj_met_sorted_list(adj, circ_met="num_emit", count=1000):
    """
    Finds as many as "count" number of isomorphic graphs out of the input one, if possible, and returns both a list of
    the adjacency matrices of those graph and a list containing the specified metric value corresponding to each of
    the graphs in the same order. The graph in the list are sorted from the minimum to maximum based on the metric value.
    :param adj: The adjacency matrix of the initial graph
    :type adj: numpy.ndarray
    :param circ_met: A metric of the quantum circuits used in graph state generation. Default value is the number of
    emitter qubits in the circuit. Supported metrics are: number of emitter "num_emit", number of CNOT gates
    between emitters "cnot", number of CNOTs per number of photons and per number of emitters in the circuit
    "cnot_per_photon" and "cnot_per_emitter", maximum depth of the circuit "depth".
    :type circ_met: str
    :param count: The number of isomorph graphs one needs to make by relabeling the initial one.
    :type count: int
    :return: Two lists. The adjacency matrices list and the metric values list.
    :rtype: list, list
    """
    if circ_met == "num_emit":
        adj_list, met_list = adj_emit_sorted_list(adj, count=count)
    else:
        graph_corr_obj = Graph_corr(num_isomorph=count, circ_metric=circ_met, initial_graph=nx.from_numpy_array(adj))
        graph_list = graph_corr_obj.graph_list
        val_list = []
        for i, graph in enumerate(graph_list):
            val = graph_corr_obj._circ_met_value(graph)
            val_list.append((i, val))
        sorted_vals = sorted(val_list, key=lambda x: x[1])
        adj_list = [nx.to_numpy_array(graph_list[x[0]]) for x in sorted_vals]
        met_list = [x[1] for x in sorted_vals]
    return adj_list, met_list


def avg_maker(x_data, y_data):
    """
    For all data points (x, y) where x is in x_data and y in y_data, finds the average and the standard deviation of all
    the points with the same x value. Returns the list of unique x values, the list of corresponding average y values,
    and the list of corresponding standard deviations of all y for each unique x.
    :param x_data: A set of numeric data
    :type x_data: An iterable
    :param y_data: A set of numeric data the same size as the x_data
    :type y_data: An iterable
    :return: Three lists. The list of unique values in the first set x_data, and the average of the respective values in
     the second set y_data, and a third list containing the standard deviation of the respective y values for each
     unique x.
    """
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


def _rep_counter(x_data, y_data):
    """
    Repetition counter. Obtains the repetition count of each point (x, y) for x in x_data and y in y_data.
    :param x_data: A set of numeric data
    :type x_data: An iterable
    :param y_data: A set of numeric data the same size as the x_data
    :type y_data: An iterable
    :return: Returns all respective x, y  value in two lists removing the redundancies and a third list containing the
    repetition count of each of the data points.
    """
    assert len(x_data) == len(y_data), "two data sets must be of same size"
    xy_list = [(x_data[i], y_data[i]) for i in range(len(x_data))]
    xy_set = set(xy_list)
    count_array = np.array([xy_list.count(xy) for xy in xy_set])
    x_list = [xy[0] for xy in xy_set]
    y_list = [xy[1] for xy in xy_set]
    return x_list, y_list, count_array


# %% node based

class Node_corr:
    """
    A class to figure out whether having a correlation between node-based graph metrics and the order of photon emission
     would affect a particular circuit metric or not. E.g. if we set the emission ordering of photons the same as/ or
     close to the ranking of their node-connectivity in the graph, would we see a decrease/increase in the depth of the
     circuit compared to a random ordering. Different type of graph and circuit metrics can be considered. The analysis
     is based on taking samples form the list of all possible node label permutations (different emission ordering for
     photons) of a graph.
    """
    def __init__(self, graph, relabel_trials=None, metric=None, circ_met="num_emit"):
        """
        Initialize a node correlation object for a specific graph (state), number of relabeling, and metrics to be used.
        :param graph: the graph at study to find correlations for
        :type graph: networkx.Graph
        :param relabel_trials: number of different node label permutations (ordering) to be tested to find correlations.
        :type relabel_trials: int
        :param metric: the main graph metric to be used; options are: Betweenness: "bet", Closeness: "close", Eigenvalue
        : "eigen", Mean neighbors’ degree: "nei_deg", Eccentricity: "eccentric", Clustering: "cluster", PageRank:
        "pagerank", Degree: "deg".
        :param circ_met: the circuit metric to study; options are number of emitter "num_emit", number of CNOT gates
        between emitters "cnot", number of CNOTs per number of photons and per number of emitters in the circuit
        "cnot_per_photon" and "cnot_per_emitter", maximum depth of the circuit "depth".
        :type circ_met: str
        """
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
        self._circ_met = circ_met
        self.adj_list, self.met_list = adj_met_sorted_list(self._adj, circ_met=self._circ_met, count=self._relabel_trials)

    @property
    def graph(self):
        return self._graph

    @graph.setter
    def graph(self, g):
        assert isinstance(g, nx.Graph)
        self._graph = g
        self._update_data()
    @property
    def circ_met(self):
        return self._circ_met

    @circ_met.setter
    def circ_met(self, met):
        self._circ_met = met
        self._update_data()
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
        self._update_data()
    @property
    def metric(self):
        return self._metric

    @metric.setter
    def metric(self, new_met):
        assert isinstance(new_met, str)
        self._metric = new_met
        self._update_data()

    def order_corr(self):
        """
        Determines and plots the relation of average pearson correlation function ,between nodes' emission order and the
         metric-value order, and the circuit metric value. The average of the graph metric is taken for all graphs that
         have the same circuit metric value.
        :return: three lists, the list of unique metric values, the list of average correlation, the list of standard
        deviations
        :rtype: list, list, list
        """
        adj_list, circ_met_list = self.adj_list, self.met_list
        data1 = [*range(self.adj.shape[0])]
        pear_corr_list = []
        for new_adj in adj_list:
            new_g = nx.from_numpy_array(new_adj)
            node_met_dict = self.metric_node_dict(new_g)
            data2 = [node_met_dict[x] for x in data1]
            # calculate Pearson's correlation
            corr, _ = pearsonr(data1, data2)
            pear_corr_list.append(abs(corr))
        overall_corr, _ = pearsonr(circ_met_list, pear_corr_list)
        print("correlation:", overall_corr)
        # retry with average values over all cases with the same number of emitters
        uniq_met_val, avg_corrs, std_corrs = avg_maker(circ_met_list, pear_corr_list)
        overall_avg_corr, _ = pearsonr(uniq_met_val, avg_corrs)
        print("avg correlation:", overall_avg_corr)
        plt.scatter(uniq_met_val, avg_corrs)
        plt.errorbar(
            uniq_met_val,
            avg_corrs,
            yerr=std_corrs,
            fmt="bo",
            ecolor="r",
            capsize=4,
            errorevery=1,
            capthick=2,
        )
        plt.ylabel(f"average emission-metric correlation\nmetric:{self.metric}")
        plt.xlabel(f"{self.circ_met}")
        plt.title(f"emission-{self.metric} correlation vs {self.circ_met}")
        plt.show()
        return uniq_met_val, avg_corrs, std_corrs

    def met_order_error(self):
        """
        Determines and plots the sum squared error between the metric and emission ordering for all nodes for a
        sample of isomorph graphs. For instance, if the ranking of nodes based on metric is the same as the emission
        ordering the error would be zero. The sum squared error is plotted vs the circuit metric for each graph. The
        correlation would be visible if small-error data points are concentrated at some specific circuit metric values.
        :return: three lists, the list of unique metric values, the list of sum squared errors, the list of repetition
        count for each of the data points over the whole sample.
        :rtype: list, list, list
        """
        adj_list, circ_met_list = self.adj_list, self.met_list
        n = self.adj.shape[0]
        err_list = []
        for new_adj in adj_list:
            new_g = nx.from_numpy_array(new_adj)
            sorted_nodes = self._metric_sorted_nodes(new_g)
            sum_sqrd_err = sum([(sorted_nodes[i] - i) ** 2 for i in range(n)])
            err_list.append(sum_sqrd_err)
        uniq_met_val, sqrd_err, count = _rep_counter(circ_met_list, err_list)
        plt.scatter(uniq_met_val, sqrd_err, s=10 * count)
        plt.title(f"emission order vs {self.metric} order Error\nfor different {self.circ_met}")
        plt.ylabel(f"sum squared difference\nmetric:{self.metric}")
        plt.xlabel(f"{self.circ_met}")
        plt.show()
        return uniq_met_val, sqrd_err, count

    def next_node_corr(self):
        """
        Determines and plots the sum squared error between the metric and emission ordering for all nodes for a
        sample of isomorph graphs. However, here the metric rank of each node is evaluated in a step by step basis when
        all other nodes that are emitted are removed from the graph. For instance, we start from the first node based on
         its emission label, we determine its metric rank (the number of nodes in the whole graph that have a lower
         metric value than the node at study). If the metric rank and order were totally correlated we would find that
          the metric rank is equal to zero, meaning that the first emitted photon has the lowest metric value. Before
          the next step we remove the first node from the graph. Next steps are carried out the same with the error
          being equal to the metric value of the node to be emitted next. (error = metric_rank - emission_rank, but
          emission rank is zero since all previous nodes are now removed)
        This is a measure of how accurate we can determine the next node to be emitted by choosing the minimum/maximum
        value node in the remaining part of the graph.
        :return: three lists, the list of unique metric values, the list of sum squared errors, the list of repetition
        count for each of the data points over the whole sample.
        :rtype: list, list, list
        """
        # based on discarding the part of the graph state that has been emitted so far.
        adj_list, circ_met_list = self.adj_list, self.met_list
        err_list = []
        for new_adj in adj_list:
            new_g = nx.from_numpy_array(new_adj)
            err_list.append(self._next_node_error(new_g))
        uniq_met_val, sqrd_err, count = _rep_counter(circ_met_list, err_list)
        plt.scatter(uniq_met_val, sqrd_err, s=10 * count)
        plt.title(f"emission order vs {self.metric}_rank Error\nfor different {self.circ_met}")
        plt.ylabel(f"sum squared difference\nmetric:{self.metric}")
        plt.xlabel(f"{self.circ_met}")
        plt.show()
        return uniq_met_val, sqrd_err, count

    def _metric_sorted_nodes(self, g):
        """
        helper function to sort nodes of the graph g based on a graph metric
        :param g: input graph
        :type g: nx.Graph
        :return: a sorted list of nodes based on metric values
        :rtype: list
        """
        node_met_dict = self.metric_node_dict(g)
        node_met_list = list(node_met_dict.items())
        met_sorted_tuples = sorted(node_met_list, key=lambda x: x[1])
        met_sorted_nodes = [x[0] for x in met_sorted_tuples]
        return met_sorted_nodes

    def _next_node_error(self, g):
        """
        helper function to determine the error value of the graph g based on the node removing method and choosing the
        next node to be emitted to be equal to the min/max value node among the remaining subgraph that is not emitted
        yet.
        :param g: input graph
        :type g: nx.Graph
        :return: the sum of squared error values of all nodes in the graph g
        :rtype: int
        """
        sum_squared = 0
        for i in range(g.number_of_nodes() - 1):
            sorted_nodes = self._metric_sorted_nodes(g)
            error = sorted_nodes.index(i)
            sum_squared += error ** 2
            g.remove_node(i)
        return sum_squared

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

    def _update_data(self):
        """
        this helper function updates the sorted list of graphs and their corresponding metric values whenever the number
         of graphs in sample, the metric, or the initial input graph itself is altered by the user.
        :return: nothing
        :rtype: None
        """
        self.adj_list, self.met_list = adj_met_sorted_list(self._adj, circ_met=self._circ_met,
                                                           count=self._relabel_trials)



# %%
class Graph_corr:
    def __init__(
            self,
            graph_metric=None,
            circ_metric=None,
            initial_graph=None,
            graph_list=None,
            trials=None,
            relabel_trials=None,
            num_isomorph=1,
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
            self._graph_metric = "node"
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
            if graph_list:
                assert isinstance(
                    graph_list[0], nx.Graph
                ), "graph_list must be a non-empty list of networkx graphs"
        else:
            self._initial_graph = initial_graph
            assert isinstance(
                initial_graph, nx.Graph
            ), "initial graph must be a networkx graph object"
            self.graph_list = self._graph_list_maker(
                self._initial_graph, count=self._num_isomorph
            )

    @property
    def initial_graph(self):
        return self._initial_graph

    @initial_graph.setter
    def initial_graph(self, g):
        assert isinstance(g, nx.Graph) or (
                g is None
        ), "initial graph must be a networkx graph object"
        self._initial_graph = g
        self.graph_list = self._graph_list_maker(
            self._initial_graph, count=self._num_isomorph
        )

    @property
    def num_isomorph(self):
        return self._num_isomorph

    @num_isomorph.setter
    def num_isomorph(self, num):
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

    @staticmethod
    def _rnd_graph_list_maker(count, n=None, p=None, n_limit=(5, 99)):
        # makes a list of relabeled graphs out of the initial one. List size = count
        graph_list = []
        if n is None:
            n_list = [random.randint(n_limit[0], n_limit[1]) for iter in range(count)]
        else:
            n_list = count * [n]
        if p is None:
            p_list = [random.randint(5, 95) / 100 for iter in range(count)]
        else:
            p_list = count * [p]

        for i in range(count):
            graph_list.append(rnd_graph(n_list[i], p_list[i]))

        return graph_list

    def finder(self, graph_type=None, show_plot=True, n=None, p=None, swap_axes=False):
        """check correlation between one graph and one circuit metric values for a set of graphs. For correlation
        between two graph or circuit metrics try the met_met method. """
        graph_met_list = []
        circ_met_list = []
        if n is None or p is None:
            self.graph_list = self._rnd_graph_list_maker(count=self.trials, n=n, p=p, n_limit=(5, 99))
        if self.graph_list is None:
            for i in range(self.trials):
                g = rnd_graph(n, p, model=graph_type)
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
        if swap_axes:
            temp = [x for x in graph_met_list]
            graph_met_list = circ_met_list
            circ_met_list = temp
        corr, _ = pearsonr(circ_met_list, graph_met_list)
        # average case
        circ_met_uniq, avg_graph_met, std_graph_met = avg_maker(
            circ_met_list, graph_met_list
        )
        avg_corr, _ = pearsonr(circ_met_uniq, avg_graph_met)
        if show_plot:
            fig, (ax1, ax2) = plt.subplots(2, figsize=(15, 15))
            ax1.scatter(circ_met_list, graph_met_list)
            ax2.plot(circ_met_uniq, avg_graph_met)
            ax2.set_xlabel(f"{self.circ_metric}") if not swap_axes else ax2.set_xlabel(f"{self.graph_metric}")
            ax2.set_ylabel(f"average {self.graph_metric}") if not swap_axes else ax2.set_ylabel(
                f"average {self.circ_metric}")
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
            _, _, plot_pair = self.finder(n=n, p=p, show_plot=False)
            plt.plot(plot_pair[0], plot_pair[1], label=f"p={round(p, 2)}")
        plt.title(
            f"the {self.graph_metric} - {self.circ_metric} correlation for different p values"
        )
        plt.legend(loc="best")
        plt.yscale("log")
        plt.show()

    def corr_n_dependence(self, p, n_step=10, constant_np=None):
        # number of nodes starting from 5 and going up to at least 100, or possibly 10 times the chosen n_step
        # if a constant_np is determined, then it is used to adjust p values such that this condition is hold
        n_list = [*range(5, max(10 * n_step, 100) + 1, n_step)]
        if constant_np:
            assert constant_np > 1, (
                "constant_np must be number larger than 1, equal to intended average degree of "
                "each node"
            )
            p_list = [constant_np / n for n in n_list]
            assert (
                    min(p_list) > 0.05
            ), "the constant_np is too low to get connected graphs for large graph size"
        for i, n in enumerate(n_list):
            p = p_list[i] if constant_np else p
            _, _, plot_pair = self.finder(n=n, p=p, show_plot=False)
            plt.plot(plot_pair[0], plot_pair[1], label=f"n={n}")
        plt.title(
            f"the {self.graph_metric} - {self.circ_metric} correlation for different n values"
        )
        plt.legend(loc="best")
        plt.yscale("log")
        plt.show()

    def met_distribution(
            self, n=None, p=None, met="num_emit", hist_bins=False, show_plot=False
    ):
        met_list = []
        if n is None or p is None:
            self.graph_list = self._rnd_graph_list_maker(count=self.trials, n=n, p=p, n_limit=(5, 99))
        if self.graph_list is None:
            for i in range(self.trials):
                g = rnd_graph(n, p)
                met_value = self._determine_met_value(met, g)
                met_list.append(met_value)
        else:
            for g in self.graph_list:
                met_value = self._determine_met_value(met, g)
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
            if not hist_bins:

                plt.scatter(dist_dict.keys(), dist_dict.values())
                if self.graph_list is None:
                    plt.figtext(
                        0.75,
                        0.70,
                        f"n: {n}\np: {p}\navg: {round(avg, 2)}\nstd: {round(std, 2)}\ntrials: "
                        f"{'{:.0e}'.format(self.trials)}",
                    )
                else:
                    plt.figtext(
                        0.75,
                        0.70,
                        f"avg: {round(avg, 2)}\nstd: {round(std, 2)}\ntrials: "
                        f"{'{:.0e}'.format(self.trials)}",
                    )
                plt.ylim(0, max(count_percent_list) * 1.25)
                plt.xticks(range(min(met_set), max(met_set) + 1))
                plt.ylabel("percentage")
            else:
                plt.hist(met_list, density=True, bins=hist_bins)
                plt.ylabel("probability density")

            plt.title(f"{met} for a sample of graphs")
            plt.xlabel(f"{met}")
            plt.show()
        return avg, std

    def met_met(self, met1, met2, n=None, p=None, show_plot=True):
        met1_list = []
        met2_list = []
        if n is None or p is None:
            self.graph_list = self._rnd_graph_list_maker(count=self.trials, n=n, p=p, n_limit=(5, 99))
        if self.graph_list is None:
            for i in range(self.trials):
                g = rnd_graph(n, p)
                met1_value = self._determine_met_value(met1, g)
                met2_value = self._determine_met_value(met2, g)
                met1_list.append(met1_value)
                met2_list.append(met2_value)
        else:
            for g in self.graph_list:
                met1_value = self._determine_met_value(met1, g)
                met2_value = self._determine_met_value(met2, g)
                met1_list.append(met1_value)
                met2_list.append(met2_value)
        x_data, y_data, count = _rep_counter(met1_list, met2_list)
        if show_plot:
            fig = plt.figure(figsize=(8, 6), dpi=300)
            fig.tight_layout()
            plt.scatter(x_data, y_data, s=10 * count)
            plt.figtext(
                0.75,
                0.70,
                f"trials:{'{:.0e}'.format(sum(count))}",
            )

            plt.title(f"{met2} vs. {met1} for a sample of graphs")
            plt.xlabel(f"{met1}")
            plt.ylabel(f"{met2}")
            plt.xticks(range(int(min(x_data)), int(max(x_data)) + 1))
            plt.ylim(min(y_data), max(y_data) * 1.25)
            plt.show()
        return x_data, y_data, count

    def _determine_met_value(self, met, graph):
        # evaluates met not knowing whether it is a graph_met or circ_met
        met_tuple = (self.graph_metric, self.circ_metric)
        try:
            self.graph_metric = met
            met_value = self._graph_met_value(graph)
        except:
            try:
                self.circ_metric = met
                met_value = self._min_circ_met_over_relabel(graph)
            except:
                raise ValueError(
                    "Metric not found. The indicated metric should be a valid graph or circuit metric"
                )
        self.graph_metric = met_tuple[0]
        self.circ_metric = met_tuple[1]
        return met_value

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


# %% visual
# fig, (ax1, ax2) = plt.subplots(2, figsize=(15, 25))
# fig.tight_layout()
# nx.draw_networkx(g3, with_labels=1, ax=ax1, pos=nx.kamada_kawai_layout(g3))
# nx.draw_networkx(new_g, with_labels=1, ax=ax2, pos=nx.kamada_kawai_layout(new_g))
# plt.show()

# %%
import time

st = time.time()
et = time.time()

elapsed_time1 = et - st
print("Execution time without and with:", elapsed_time1)
