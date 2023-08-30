import networkx as nx
import numpy as np
from warnings import warn
from tkinter import messagebox
import tkinter as tk
import os

import src.ops as ops
import matplotlib.pyplot as plt
import src.noise.monte_carlo_noise as mcn
import src.utils.preprocessing as pre
import src.utils.circuit_comparison as comp
import src.noise.noise_models as nm
import src.backends.lc_equivalence_check as lc
import src.backends.stabilizer.functions.local_cliff_equi_check as slc
from src.solvers.evolutionary_solver import (
    EvolutionarySolver,
    EvolutionarySearchSolverSetting,
)

from src.solvers.solver_base import SolverBase
from src.solvers.deterministic_solver import DeterministicSolver
from src.backends.compiler_base import CompilerBase
from src.circuit import CircuitDAG
from src.metrics import MetricBase
from src.state import QuantumState
from src.io import IO
from src.utils.relabel_module import (
    iso_finder,
    emitter_sorted,
    lc_orbit_finder,
    get_relabel_map,
)
from src.backends.state_representation_conversion import stabilizer_to_graph
from src.backends.stabilizer.functions.rep_conversion import (
    get_clifford_tableau_from_graph,
)
from src.backends.stabilizer.compiler import StabilizerCompiler
from src.backends.density_matrix.compiler import DensityMatrixCompiler
from src.metrics import Infidelity
from src.backends.stabilizer.state import Stabilizer
from src.solvers.hybrid_solvers import HybridGraphSearchSolverSetting
from src.solvers.hybrid_solvers import AlternateGraphSolver

from benchmarks.graph_states import *
from src.data_collection.correlation_module import _rep_counter

import ray


# target graph maker
def t_graph(gtype, n, seed=None, show=False):
    """
    If g_type is lattice then "n" must be a tuple (rows, columns).
    If g_type is adj then "n" must be an adjacency matrix.
    :param gtype: the type of the graph
    :type gtype: str
    :param n: size of the graph, not equal to the actual number of nodes in some cases
    :type n: int or tuple
    :param seed: seed
    :type seed: int
    :param show: if true the graph will be drawn
    :type show: bool
    :return: the generated graph object
    :rtype: nx.Graph
    """
    if gtype == "draw":
        g = graph_gui()
    elif gtype == "tree":
        g = nx.random_tree(n, seed=seed)
    elif gtype == "rnd":
        rng = np.random.default_rng(seed=seed)
        p = rng.integers(low=15, high=95, size=1)[0] / 100
        g = nx.from_numpy_array(
            nx.to_numpy_array(random_graph_state(n, p_edge=p, np_rng=rng).data)
        )
    elif gtype == "rgs":
        g = repeater_graph_states(n)
    elif gtype == "linear":
        g = nx.from_numpy_array(nx.to_numpy_array(linear_cluster_state(n).data))
    elif gtype == "cycle":
        g = nx.from_numpy_array(nx.to_numpy_array(linear_cluster_state(n).data))
        g.add_edge(0, n - 1)
    elif gtype == "lattice":
        g = nx.from_numpy_array(
            nx.to_numpy_array(lattice_cluster_state((n[0], n[1])).data)
        )
    elif gtype == "star":
        g = nx.from_numpy_array(nx.to_numpy_array(star_graph_state(n).data))
    elif gtype == "adj":
        g = nx.from_numpy_array(n)
    elif gtype == "nx":
        # networkx graph
        assert isinstance(n, nx.Graph)
        g = n
    else:
        raise ValueError
    if show:
        nx.draw_networkx(
            g,
            with_labels=1,
            pos=nx.kamada_kawai_layout(g) if gtype != "lattice" else None,
        )
        plt.show()
    return g


class InputParams:
    def __init__(
        self,
        n_ordering=10,  # input
        rel_inc_thresh=0.2,  # advanced: (0,1) The closer to 0 the closer we get to an exhaustive search for reordering.
        allow_exhaustive=True,  # advanced*: only reason to deactivate is to save runtime if this is the bottleneck
        iso_thresh=None,  # advanced: if not enough relabeled graphs are found, set it to a larger number!
        n_lc_graphs=10,  # input
        lc_orbit_depth=None,  # advanced: if hit the runtime limit, limit len(sequence of Local complementations)
        lc_method=None,
        noise_simulation=True,  # input
        noise_model_mapping="depolarizing",  # input
        depolarizing_rate=0.01,  # input
        error_margin=0.01,  # input
        confidence=0.95,  # input
        mc_map=None,  # advanced*: pass a manual noise map for the monte carlo simulations
        n_cores: int = 8,  # advanced: change if processor has different number of cores
        seed=None,  # input
        graph_type="rnd",  # input
        graph_size=5,  # input
        verbose=False,
        save_openqasm: str = "none",
    ):
        # other parameters
        self.number_of_cores = n_cores
        self.err_margin = error_margin
        self.conf_level = confidence
        self.seed = seed
        self.monte_carlo_noise_map = mc_map
        # target graph
        self.graph_type = graph_type
        self.graph_size = graph_size
        self.target_graph = t_graph(self.graph_type, self.graph_size, self.seed)
        # setting
        self.setting = HybridGraphSearchSolverSetting()
        # setting options
        self.setting.allow_relabel = bool(n_ordering)
        self.setting.n_iso_graphs = n_ordering
        self.setting.rel_inc_thresh = rel_inc_thresh
        self.setting.allow_exhaustive = allow_exhaustive
        self.setting.iso_thresh = iso_thresh
        self.setting.n_lc_graphs = n_lc_graphs
        self.setting.lc_orbit_depth = lc_orbit_depth
        self.setting.lc_method = lc_method
        self.setting.depolarizing_rate = depolarizing_rate
        self.setting.verbose = verbose
        self.setting.save_openqasm = save_openqasm
        # solver
        self.auto_noise_params()  # auto assign noise parameters to setting
        self.solver = AlternateGraphSolver(
            target_graph=self.target_graph,
            graph_solver_setting=self.setting,
            noise_model_mapping=noise_model_mapping,
        )
        # solver options
        self.solver.metric = Infidelity
        self.solver.compiler = StabilizerCompiler()
        self.solver.noise_compiler = DensityMatrixCompiler()

        self.solver.io = None
        self.solver.noise_simulation = noise_simulation
        self.solver.seed = seed

    def auto_noise_params(self):
        if self.err_margin == 0 or self.conf_level == 1 or len(self.target_graph) <= 5:
            self.setting.monte_carlo = False
        else:
            self.setting.monte_carlo = True
            n_total = (
                int(0.5 * np.log(2 / (1 - self.conf_level)) * self.err_margin ** (-2))
                + 1
            )
            n_single = int(n_total / self.number_of_cores) + 1
            n_parallel = self.number_of_cores
            self.setting.monte_carlo_params = {
                "n_sample": n_total,
                "map": mcn.McNoiseMap(),
                "compiler": StabilizerCompiler(),
                "seed": self.seed,
                "n_parallel": n_parallel,
                "n_single": n_single,
            }
            if n_total > 10e6:
                warn(
                    f"The Monte Carlo {n_total} runs may take too long. Consider decreasing the accuracy"
                )
        return


def input_gui():
    input_params = []

    def on_entry_focus_in(event):
        """
        Function to handle the <FocusIn> event on the Entry widget.
        """
        entry = root.focus_get()
        # Check if the text color is still grey
        if entry.cget("foreground") == "grey":
            entry.delete(0, tk.END)  # Remove the default value
            entry.config(fg="black")  # Change text color to black

    def toggle_advanced_options():
        """
        Function to toggle the visibility of advanced options.
        """
        if advanced_options_frame.winfo_ismapped():
            advanced_options_frame.grid_remove()
        else:
            advanced_options_frame.grid(row=root.grid_size()[1], column=0, sticky=tk.W)

    def submit_form():
        """
        Function to handle form submission.
        """
        params_val = []
        param1_val = param1_entry.get()
        param2_val = param2_entry.get()
        param3_val = param3_entry.get()
        param4_val = param4_entry.get()
        print("Param 1: ", param1_val)
        print("Param 2: ", param2_val)
        print("Param 3: ", param3_val)
        print("Param 4: ", param4_val)
        for param in input_params:
            params_val.append(param.get())
        return params_val

    # Create main window
    root = tk.Tk()
    root.title("Parameter Form")
    main_frame = tk.Frame(root)
    # Create input widgets
    param1_label = tk.Label(main_frame, text="Param 1:")
    param1_entry = tk.Entry(main_frame, width=30)
    param1_entry.insert(0, 111)  # Set default value
    param1_entry.config(fg="grey")  # Set text color to grey
    param2_label = tk.Label(main_frame, text="Param 2:")
    param2_entry = tk.Entry(main_frame, width=30)
    param2_entry.insert(0, "Default Value 2")  # Set default value
    param2_entry.config(fg="grey")  # Set text color to grey

    # Create frame for advanced options
    advanced_options_frame = tk.Frame(root)
    # Create input widgets for advanced options
    param3_label = tk.Label(advanced_options_frame, text="Param 3:")
    param3_entry = tk.Entry(advanced_options_frame, width=30)
    param3_entry.insert(0, "Default Value 3")  # Set default value
    param3_entry.config(fg="grey")  # Set text color to grey
    param4_label = tk.Label(advanced_options_frame, text="Param 4:")
    param4_entry = tk.Entry(advanced_options_frame, width=30)
    param4_entry.insert(0, "Default Value 4")  # Set default value
    param4_entry.config(fg="grey")  # Set text color to grey

    input_params = [param1_entry, param2_entry, param3_entry, param4_entry]
    # Bind <FocusIn> event to entry widget
    for i, entries in enumerate(input_params):
        entries.bind("<FocusIn>", on_entry_focus_in)

    advanced_options_button = tk.Button(
        main_frame, text="Advanced Options", command=toggle_advanced_options
    )
    submit_button = tk.Button(main_frame, text="Submit", command=submit_form)

    # Pack input widgets
    # param1_label.pack(side=tk.LEFT)
    # param1_entry.pack(side=tk.TOP)
    # param2_label.pack(side=tk.BOTTOM)
    # param2_entry.pack(side=tk.LEFT)
    # # Pack input widgets for advanced options
    # param3_label.pack(side=tk.LEFT)
    # param3_entry.pack(side=tk.TOP)
    # param4_label.pack(side=tk.LEFT)
    # param4_entry.pack(side=tk.TOP)

    # Grid layout
    main_frame.grid(row=0, column=0, sticky=tk.W)
    param1_label.grid(row=0, column=0, sticky=tk.W)
    param1_entry.grid(row=0, column=1)
    param2_label.grid(row=1, column=0, sticky=tk.W)
    param2_entry.grid(row=1, column=1)
    # advanced ones
    param3_label.grid(row=2, column=0, sticky=tk.W)
    param3_entry.grid(row=2, column=1)
    param4_label.grid(row=3, column=0, sticky=tk.W)
    param4_entry.grid(row=3, column=1)

    advanced_options_button.grid(row=0, column=2, sticky=tk.W)
    submit_button.grid(row=1, column=2, sticky=tk.W)

    root.focus_set()
    root.mainloop()


def graph_gui():
    class Point:
        def __init__(self, x, y):
            self.x = x
            self.y = y

    class Graph:
        def __init__(self):
            self.points = []
            self.nodes = set()
            self.edges = []
            self.edge_list = []
            self.is_stopped = False
            self.p_node_dict = dict()

        def add_point(self, x, y):
            p = Point(x, y)
            self.points.append(p)
            n = (x, y)
            if n not in self.nodes:
                self.nodes.add(n)
                self.p_node_dict[n] = len(self.nodes) - 1

        def add_edge(self, p1, p2):
            self.edges.append((p1, p2))
            self.edge_list.append(
                (self.p_node_dict[(p1.x, p1.y)], self.p_node_dict[(p2.x, p2.y)])
            )

        def reset(self):
            self.points = []
            self.nodes = set()
            self.edges = []
            self.edge_list = []
            self.p_node_dict = dict()
            self.is_stopped = True

        def print_graph(self):
            print("Points:")
            for point in self.points:
                print(f"({point.x}, {point.y})")
            print("Edges:")
            for edge in self.edges:
                print(f"({edge[0].x}, {edge[0].y}) - ({edge[1].x}, {edge[1].y})")

    def create_point(event):

        x = round(event.x / GRID_SIZE) * GRID_SIZE
        y = round(event.y / GRID_SIZE) * GRID_SIZE
        graph.add_point(x, y)
        canvas.create_oval(
            x - POINT_SIZE,
            y - POINT_SIZE,
            x + POINT_SIZE,
            y + POINT_SIZE,
            fill=POINT_COLOR,
        )

        # Check if there are at least two points to create an edge
        if graph.is_stopped:
            virtual_length = len(graph.points) - 1
            graph.is_stopped = False
        else:
            virtual_length = 0
        if len(graph.points) - virtual_length >= 2:
            p1 = graph.points[-2]
            p2 = graph.points[-1]
            graph.add_edge(p1, p2)
            canvas.create_line(p1.x, p1.y, p2.x, p2.y, fill=EDGE_COLOR)

    def reset_graph():
        graph.reset()
        canvas.delete("all")
        # Draw the initial grid
        for x in range(0, 400, GRID_SIZE):
            canvas.create_line(x, 0, x, 400, fill="gray", dash=(4, 4))
        for y in range(0, 400, GRID_SIZE):
            canvas.create_line(0, y, 400, y, fill="gray", dash=(4, 4))

    def stop_graph():
        graph.is_stopped = True

    # Constants for grid size and point size
    GRID_SIZE = 20
    POINT_SIZE = 5

    # Constants for point and edge colors
    POINT_COLOR = "red"
    EDGE_COLOR = "blue"

    # Create the main window
    root = tk.Tk()
    root.title("Graph Creator")

    # Create a canvas for drawing points and edges
    canvas = tk.Canvas(root, width=400, height=400, bg="white")
    canvas.pack()
    canvas.bind("<Button-1>", create_point)

    # Create a reset button
    btn_reset = tk.Button(root, text="Reset", command=reset_graph)
    btn_reset.pack()

    # Create a stop button
    btn_stop = tk.Button(root, text="Next", command=stop_graph)
    btn_stop.pack()

    # Create a Graph object to store points and edges
    graph = Graph()

    # Draw the initial grid
    for x in range(0, 400, GRID_SIZE):
        canvas.create_line(x, 0, x, 400, fill="gray", dash=(4, 4))
    for y in range(0, 400, GRID_SIZE):
        canvas.create_line(0, y, 400, y, fill="gray", dash=(4, 4))

    # Start the GUI event loop
    root.mainloop()

    g = nx.Graph()
    g.add_edges_from(graph.edge_list)
    return g


def graph_met_value(graph_metric, g):
    """
    Evaluates the graph metric for the given graph.
    :param g: graph at study
    :type g: nx.Graph
    :return: the graph metric value
    :rtype: int or float
    """
    if graph_metric == "max_between":
        dict_centrality = nx.betweenness_centrality(g)
        graph_value = max(dict_centrality.values())
    elif graph_metric == "max_close":
        dict_centrality = nx.closeness_centrality(g)
        graph_value = max(dict_centrality.values())
    elif graph_metric == "min_close":
        dict_centrality = nx.closeness_centrality(g)
        graph_value = min(dict_centrality.values())
    elif graph_metric == "mean_nei_deg":
        # the mean of the "average neighbors degree" over all nodes in graph
        dict_met = nx.average_neighbor_degree(g)
        graph_value = np.mean(list(dict_met.values()))
    elif graph_metric == "max_deg":
        dict_met = dict(g.degree())
        graph_value = max(list(dict_met.values()))
    elif graph_metric == "node_connect":
        graph_value = nx.node_connectivity(g)
    elif graph_metric == "edge_connect":
        graph_value = nx.edge_connectivity(g)
    elif graph_metric == "assort":
        graph_value = nx.degree_assortativity_coefficient(g)
    elif graph_metric == "radius":
        graph_value = nx.radius(g)
    elif graph_metric == "diameter":
        graph_value = nx.diameter(g)
    elif graph_metric == "periphery":
        # num of nodes with distance equal to diameter
        graph_value = len(nx.periphery(g))
    elif graph_metric == "center":
        # num of nodes with distance equal to radius
        graph_value = len(nx.center(g))
    elif graph_metric == "cluster":
        graph_value = nx.average_clustering(g)
    elif graph_metric == "local_efficiency":
        graph_value = nx.local_efficiency(g)
    elif graph_metric == "global_efficiency":
        graph_value = nx.global_efficiency(g)
    elif graph_metric == "node":
        graph_value = g.number_of_nodes()
    elif graph_metric == "avg_shortest_path":
        graph_value = nx.average_shortest_path_length(g)
    elif graph_metric == "n_edges":
        graph_value = nx.number_of_edges(g)
    elif graph_metric == "pop":
        nodes = g.number_of_nodes()
        edges = g.size()
        graph_value = edges / ((nodes * (nodes - 1)) / 2)
    else:
        raise ValueError(
            f"Graph metric {graph_metric} not found. It may not be implemented"
        )

    return graph_value


def met_hist(result, met, show_plot=True, store=True, index_list=None, dir_name="new"):
    values = _met2val(result, met, index_list=index_list)
    fig = plt.figure(figsize=(8, 6), dpi=300)
    n, bins, patches = plt.hist(
        x=values, bins="auto", color="#0504aa", alpha=0.7, rwidth=0.85
    )
    plt.grid(axis="y", alpha=0.75)
    plt.xlabel(f"{met} Value")
    plt.ylabel("Frequency")
    plt.title(f"{met} distribution")
    # plt.text(23, 45, r'$\mu=15, b=3$')
    # plt.figtext(0.75, 0.70, "foo")
    maxfreq = n.max()
    # y axis upper limit
    plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
    if store:
        new_path = f"/Users/sobhan/Desktop/demo storage/{dir_name}"
        if not os.path.exists(new_path):
            os.makedirs(new_path)
        plt.savefig(new_path + f"/{met}_distribution.png")
    if show_plot:
        plt.show()
    else:
        plt.close()


def met_met(
    result, met1, met2, show_plot=True, store=True, index_list=None, dir_name="new"
):
    met1_list = _met2val(result, met1, index_list=index_list)
    met2_list = _met2val(result, met2, index_list=index_list)
    fig = plt.figure(figsize=(8, 6), dpi=300)
    x_data, y_data, count = _rep_counter(met1_list, met2_list)
    plt.scatter(x_data, y_data, s=15 * count)
    # plt.figtext(0.75, 0.70, "foo")
    plt.title(f"{met2} vs. {met1} correlation")
    plt.xlabel(f"{met1}")
    plt.ylabel(f"{met2}")
    if store:
        new_path = f"/Users/sobhan/Desktop/demo storage/{dir_name}"
        if not os.path.exists(new_path):
            os.makedirs(new_path)
        plt.savefig(new_path + f"/{met1}_{met2}.png")
    if show_plot:
        plt.show()
    else:
        plt.close()


def _met2val(result, met, index_list=None):
    g_met = False  # is it a graph metric?
    index_range = index_list if index_list else range(len(result))

    if "graph_metric" in result._data:
        if met in result["graph_metric"][0]:
            values = [result["graph_metric"][i][met] for i in index_range]
            g_met = True
    if not g_met:
        assert (
            met in result._data
        ), f"the metric: {met} is not calculated in the results"
        values = [result[met][i] for i in index_range]
    return values
