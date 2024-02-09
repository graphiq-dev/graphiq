import os
import glob
import pandas as pd
import networkx as nx
import ast
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from qiskit import QuantumCircuit
from src.data_collection.ui_functions import *

from src.solvers.deterministic_solver import DeterministicSolver
from src.circuit import CircuitDAG
from src.state import QuantumState
from src.utils.relabel_module import *
from src.backends.stabilizer.functions.rep_conversion import (
    get_clifford_tableau_from_graph,
)
from src.backends.stabilizer.compiler import StabilizerCompiler
from src.metrics import Infidelity
from scipy.optimize import curve_fit
from src.data_collection.user_interface import result_maker


def txt2txt():
    # specify the directory
    dir_path = f'/Users/sobhan/Desktop/EntgClass1'

    # specify the output file
    out_file = f'/Users/sobhan/Desktop/EntgClass1/all.txt'

    with open(out_file, 'w') as outfile:
        # walk through the directory
        for dirpath, dirnames, filenames in os.walk(dir_path):
            # find all 'bests.txt' files in the current directory
            for filename in glob.glob(os.path.join(dirpath, 'bests.txt')):
                # write the name of the subdirectory (header)
                outfile.write(f'Header: {os.path.basename(dirpath)}\n')

                with open(filename, 'r') as infile:
                    lines = infile.readlines()
                    # write the first line
                    outfile.write(lines[0])

                    # write the last line
                    # but we need to make sure there's more than one line
                    if len(lines) > 1:
                        outfile.write(lines[-1])

                # add a newline for separation
                outfile.write('\n')


def min_max_indices(lst):
    max_val = max(lst)
    min_val = min(lst)

    max_indices = [i for i, x in enumerate(lst) if x == max_val]
    min_indices = [i for i, x in enumerate(lst) if x == min_val]

    return (max_val, max_indices), (min_val, min_indices)


def csv_to_dict(filepath):
    # read the DataFrame from csv
    df = pd.read_csv(filepath)

    # convert the DataFrame to dictionary
    out_dict = df.to_dict(orient='list')  # or 'series', 'split', 'records', 'index' depending on your needs

    return out_dict


# %%
def back2dict(class_n=1, case_n=0):
    filepath = f'/Users/sobhan/Desktop/EntgClass1/class {class_n}/case{case_n}.csv'  # replace with your file path
    my_dict = csv_to_dict(filepath)
    cost = []
    cnot_cost = []  # prioritize cnot count then emitters then depth
    depth_cost = []  # prioritize eff depth then emitters then cnots
    for i in range(len(my_dict['n_emitters'])):
        cost.append(my_dict['n_emitters'][i] * 10000 + my_dict['n_cnots'][i] * 100 + my_dict['max_emit_eff_depth'][i])
        cnot_cost.append(
            my_dict['n_emitters'][i] * 100 + my_dict['n_cnots'][i] * 10000 + my_dict['max_emit_eff_depth'][i])
        depth_cost.append(
            my_dict['n_emitters'][i] * 100 + my_dict['n_cnots'][i] + my_dict['max_emit_eff_depth'][i] * 10000)
    my_dict['cost'] = cost
    my_dict['cnot_cost'] = cnot_cost
    my_dict['depth_cost'] = depth_cost
    # print("max cost and min cost and their indices", min_max_indices(my_dict['cost']))
    return my_dict


def back2graphs(n_class, number_of_cases=21):
    graph_dict = {}
    for i in range(number_of_cases):
        ddd = back2dict(class_n=n_class, case_n=i)
        graphs = [edgelist2graph(ast.literal_eval(edges)) for edges in ddd['edges']]
        graph_dict[f'case{i}'] = graphs
    return graph_dict


# %%
def graph_to_circ(graph, noise_model_mapping=None, show=False):
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
    c_tableau = get_clifford_tableau_from_graph(graph)
    ideal_state = QuantumState(n, c_tableau, representation="stabilizer")

    target = ideal_state
    solver = DeterministicSolver(
        target=target,
        metric=Infidelity(target),
        compiler=StabilizerCompiler(),
        noise_model_mapping=noise_model_mapping,
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


# %%
def circ_plot(c_list, grid=(2, 1), size=(6, 8), dpi=200, output="mpl"):
    n = len(c_list)
    fig = plt.figure(figsize=size, dpi=dpi)
    ax_list = []
    out = []
    display_text = c_list[0].openqasm_symbols
    style = {"displaytext": display_text}
    style["figwidth"] = 20
    style["dpi"] = dpi

    for i in range(1, n + 1):
        axs = fig.add_subplot(grid[0], grid[1], i)
        ax_list.append(axs)
    for i, c in enumerate(c_list):
        qc = QuantumCircuit.from_qasm_str(c.to_openqasm())
        out.append(qc.draw(output=output, ax=ax_list[i], plot_barriers=False, style=style))
        # c.draw_circuit(ax=ax_list[i], show=False)
    plt.show()
    return out


# %%


def graph2fid(g, error_margin=0.01, confidence=0.99):
    user_input = InputParams(
        n_ordering=1,  # input
        rel_inc_thresh=0.2,  # advanced: (0,1) The closer to 0 the closer we get to an exhaustive search for reordering.
        allow_exhaustive=True,  # advanced*: only reason to deactivate is to save runtime if this is the bottleneck
        iso_thresh=None,  # advanced: if not enough relabeled graphs are found, set it to a larger number!
        n_lc_graphs=1,  # input
        lc_orbit_depth=1,  # advanced: if hit the runtime limit, limit len(sequence of Local complementations)
        lc_method=None,  # input
        noise_simulation=True,  # input
        noise_model_mapping="depolarizing",  # input
        depolarizing_rate=0.005,  # input
        error_margin=error_margin,  # input
        confidence=confidence,  # input
        mc_map=None,  # advanced*: pass a manual noise map for the monte carlo simulations
        n_cores=8,  # advanced: change if processor has different number of cores
        seed=1,  # input
        graph_type="nx",  # input
        graph_size=g,  # input
        verbose=False,
        save_openqasm="none")
    solver = user_input.solver
    settings = user_input.setting
    solver.solve()
    result = solver.result
    out_dict = (result._data).copy()
    out_dict['fidelity'] = [round(1 - s, 5) for s in out_dict['score']]
    return out_dict['fidelity']


# %%
def params_finder(e_list):
    orbit_list = [edgelist2graph(edge_list) for edge_list in e_list]
    cs = []
    for g in orbit_list:
        cs.append(graph_to_circ(g))

    unwrapped_circ = [c.copy() for c in cs]
    max_emit_depth = []
    max_emit_eff_depth = []
    depth = []
    n_emitters = []
    n_cnots = []
    for c in unwrapped_circ:
        c.unwrap_nodes()
        c.remove_identity()
        # calculate emitter depths
        e_depth = {}
        eff_depth = {}
        for e_i in range(c.n_emitters):
            e_depth[e_i] = len(c.reg_gate_history(reg=e_i)[1]) - 2
            # find the max topological depth between two consecutive measurements on the same emitter
            node_list = []
            for i, oper in enumerate(c.reg_gate_history(reg=e_i)[0]):
                # first find a list of nodes in DAG corresponding to measurements
                if type(oper).__name__ in ['Input', 'MeasurementCNOTandReset', 'Output']:
                    node_list.append(c.reg_gate_history(reg=e_i)[1][i])
            node_depth_list = [c._max_depth(n) for n in node_list]
            depth_diff = [node_depth_list[j + 1] - node_depth_list[j] for j in range(len(node_list) - 1)]
            eff_depth[e_i] = max(depth_diff)
        max_emit_depth.append(max(e_depth.values()))
        max_emit_eff_depth.append(max(eff_depth.values()))
        depth.append(max(c.register_depth["e"]))
        # calculate n_emitter and n_cnots
        n_emitters.append(c.n_emitters)
        if "Emitter-Emitter" in c.node_dict:
            n_cnots.append(len(c.get_node_by_labels(["Emitter-Emitter", "CNOT"])))
        else:
            n_cnots.append(0)

    for x in [max_emit_depth, max_emit_eff_depth, depth, n_emitters, n_cnots]:
        df = pd.DataFrame(x)
        fpath = f'/Users/sobhan/Desktop/{x}.csv'
        df.to_csv(fpath, header=False, index=False)
    return [max_emit_depth, max_emit_eff_depth, depth, n_emitters, n_cnots]


# %%
# read avgs and std from csv rows
def plot_scaling(path1, path2):
    # path: f'/Users/sobhan/Desktop//EntgClass/random_8_05_three/bwDepth.csv'
    df1 = pd.read_csv(path1, header=None)
    df2 = pd.read_csv(path2, header=None)

    ll = []
    for index, row in df1.iterrows():
        row_list = row.tolist()
        ll.append(row_list)
    ll2 = []
    for index, row in df2.iterrows():
        row_list = row.tolist()
        ll2.append(row_list)
    if isinstance(ll[0][0], str):
        cnots = [ast.literal_eval(x) for x in ll[0]]
        cstds = [ast.literal_eval(x) for x in ll[1]]
    else:
        cnots = [(x) for x in ll[0]]
        cstds = [(x) for x in ll[1]]
    cnot_range = [x[1] - x[0] for x in cnots]
    cstd_range = [np.sqrt(x[0] ** 2 + x[1] ** 2) for x in cstds]
    x_perc = [*range(1, 10)] + [*range(10, 100, 10)]
    x_perc = [i / 644 * 100 for i in x_perc]
    y_perc = [i / 7 * 100 for i in cnot_range]
    y_std_perc = [i / 7 * 100 for i in cstd_range]

    depths = [ast.literal_eval(x) for x in ll2[0]]
    dstds = [ast.literal_eval(x) for x in ll2[1]]
    depth_range = [x[1] - x[0] for x in depths]
    dstd_range = [np.sqrt(x[0] ** 2 + x[1] ** 2) for x in dstds]
    x2_perc = [*range(1, 10)] + [*range(10, 100, 10)]
    x2_perc = [i / 644 * 100 for i in x2_perc]
    y2_perc = [i / 32 * 100 for i in depth_range]
    y2_std_perc = [i / 32 * 100 for i in dstd_range]

    fig, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.set_xlabel('LC %')
    ax1.set_ylabel('CNOT range %', color=color)
    ax1.errorbar(x_perc, y_perc, yerr=y_std_perc, fmt='-o', color=color, ecolor='green', capsize=2)
    ax1.tick_params(axis='y', labelcolor=color)
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax2.set_ylabel('Depth range %', color=color)  # we already handled the x-label with ax1
    ax2.errorbar(x_perc, y2_perc, yerr=y2_std_perc, fmt='-*', color=color, ecolor='black', capsize=2)
    ax2.tick_params(axis='y', labelcolor=color)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()


# %%
# correlation between two lists
from scipy.stats import spearmanr, pearsonr
import statsmodels.api as sm
import statsmodels.formula.api as smf


def corrs(list1, list2):
    corr_s, p_value_s = spearmanr(list1, list2)
    print('Spearmans correlation: %.3f' % corr_s, p_value_s)
    corr_p, p_value_p = pearsonr(list1, list2)
    print('Pearsons correlation: %.3f' % corr_p, p_value_p)
    return corr_s, p_value_s, corr_p, p_value_p


def curviliner(list1, list2):
    df = pd.DataFrame({'x': list1, 'y': list2})
    df['x_squared'] = df['x'] ** 2
    # Fit a quadratic model
    model = smf.ols(formula='y ~ x + x_squared', data=df)
    results = model.fit()

    print(results.summary())


# %%
# theory proof of scaling
import math
import scipy.integrate


class RangeProb:
    def __init__(self, m, s, r, k, delta_r=0.5, domain=(0, 100)):
        """
        Given the specifications a distribution numbers, here we determine the probability and the probability density
        of having a range of 'r' in a sample of size 'k'.  [r = max(sample)-min(sample)]

        :param s: standard deviation
        :param m: mean
        :param r: range
        :param delta_r: the interval around 'r' to integrate over to find the probability
        :param domain: the range of numbers considered in the integration over Gaussian distribution of the initial data
        """
        self.s = s
        self.m = m
        self.r = r
        self.k = k
        self.delta_r = delta_r
        self.domain = domain

    def gaus_prob(self, x1, x2):
        return self._erf_maker(x2) - self._erf_maker(x1)

    def _erf_maker(self, x):
        return 0.5 * math.erf((x - self.m) / (self.s * math.sqrt(2)))

    def prob_x2_x1_integrand(self, x1, r, k):
        p1 = (1 / (self.s * math.sqrt(2 * math.pi))) * math.e ** (-0.5 * ((x1 - self.m) / self.s) ** 2)
        p2r = (1 / (self.s * math.sqrt(2 * math.pi))) * math.e ** (-0.5 * ((x1 + r - self.m) / self.s) ** 2)
        p2l = (1 / (self.s * math.sqrt(2 * math.pi))) * math.e ** (-0.5 * ((x1 - r - self.m) / self.s) ** 2)
        p_12r = self.gaus_prob(x1, x1 + r) ** (k - 2)
        p_12l = self.gaus_prob(x1 - r, x1) ** (k - 2)

        return p1 * (p_12r * p2r + p_12l * p2l) * (k - 1) * k / 2

    def prob_density_r(self, r, k):
        assert k >= 2
        result, error = scipy.integrate.quad(self.prob_x2_x1_integrand, self.domain[0], self.domain[1], args=(r, k))
        return result

    def prob_r(self, k):
        result, error = scipy.integrate.quad(self.prob_density_r, self.r - self.delta_r, self.r + self.delta_r,
                                             args=(k))
        return result

    def pdf(self, r):
        return self.prob_density_r(r, self.k)


# %%
# fit normal dist to a given list of occurrence
from scipy.stats import norm
from scipy.stats import shapiro  # Shapiro goodness of the fit to normal distribution test


# %%

def list2gaus(scores, show=True, n_bins=13, full_box=False):
    fig, ax = plt.subplots(figsize=(8, 6), dpi=200)
    stat, p_shapiro = shapiro(scores)
    mu, std = norm.fit(scores)

    hist_values, bins, _ = ax.hist(scores, bins=n_bins, density=True, alpha=0.6, color='blue', edgecolor='black',
                                   align='left', rwidth=0.9)
    print(np.sum(hist_values), hist_values, bins)
    bin_middles = (bins[:-1] + bins[1:]) / 2.0
    p = norm.pdf(bin_middles, mu, std)
    total = np.sum(p)
    normalized_p = p * total
    # ax.plot(bin_middles, normalized_p, 'k', linewidth=2, linestyle='dashed')

    x = np.linspace(min(scores), max(scores), 100)
    p = norm.pdf(x, mu, std)
    ax.plot(x, p, 'k', linewidth=2, linestyle='dashed')

    text_1 = "Gaussian fit:\n$\mu =$ %.2f,  $\sigma =$ %.2f" % (mu, std)
    ax.set_xlabel('Number of CNOT gates')
    ax.set_ylabel('Relative Frequency')
    ax.set_title("Full orbit CNOT distribution")

    ax.set_ylim(0, 0.3)

    ax.text(14, .18, text_1, fontsize=12, bbox=dict(facecolor="white", alpha=0.5, edgecolor='None'))
    # Set grid
    ax.grid(False)
    # Set the font sizes
    ax.xaxis.label.set_size(14)
    ax.yaxis.label.set_size(14)
    ax.title.set_size(16)
    ax.xaxis.set_major_locator(ticker.AutoLocator())
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_major_locator(ticker.AutoLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.tick_params(axis='both', which='minor', labelsize=12)
    # Optional: remove top and right spines for a cleaner look
    if not full_box:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    if show:
        plt.show()
    return mu, std, (stat, p_shapiro), fig


def pdf2gaus(pdf):
    # pdf is the probability density function
    mean, _ = scipy.integrate.quad(lambda x: x * pdf(x), 0, np.inf)
    variance, _ = scipy.integrate.quad(lambda x: (x - mean) ** 2 * pdf(x), 0, np.inf)

    return mean, math.sqrt(variance)


# %%
def rgs_results(n):
    rgs_n = repeater_graph_states(n)
    ui_rgs_n = InputParams(
        n_ordering=1,  # input
        rel_inc_thresh=0.2,  # advanced: (0,1) The closer to 0 the closer we get to an exhaustive search for reordering.
        allow_exhaustive=True,  # advanced*: only reason to deactivate is to save runtime if this is the bottleneck
        iso_thresh=None,  # advanced: if not enough relabeled graphs are found, set it to a larger number!
        n_lc_graphs=None,  # input
        lc_orbit_depth=None,  # advanced: if hit the runtime limit, limit len(sequence of Local complementations)
        lc_method="rgs",  # input
        noise_simulation=False,  # input
        noise_model_mapping="depolarizing",  # input
        depolarizing_rate=0.005,  # input
        error_margin=0.1,  # input
        confidence=1,  # input
        mc_map=None,  # advanced*: pass a manual noise map for the monte carlo simulations
        n_cores=8,  # advanced: change if processor has different number of cores
        seed=1,  # input
        graph_type="nx",  # input
        graph_size=rgs_n,  # input
        verbose=False,
        save_openqasm="none")
    solver_rgs_n = ui_rgs_n.solver
    solver_rgs_n.solve()
    res_rgs20 = solver_rgs_n.result
    result_maker(res_rgs20, graph_met_list=["n_edges"], circ_met_list=["n_cnots", "max_emit_depth",
                                                                       "max_emit_reset_depth", "n_unitary"])
    return res_rgs20


def pos_maker(n):
    # position for the windmill graph
    n = n + 1
    pos = {0: (-0.25, 0), 1: (0.25, 0)}
    theta = np.linspace(0, 2 * np.pi, 2 * (n - 1) - 1)
    theta = theta + (theta[1] / 2)
    for i in range(1, n - 1):
        pos[2 * i] = (np.cos(theta[2 * (i - 1)]), np.sin(theta[2 * (i - 1)]))
        pos[2 * i + 1] = (np.cos(theta[2 * (i - 1) + 1]), np.sin(theta[2 * (i - 1) + 1]))
    return pos


def pos_maker_rgs_orbit(n, j):
    # n is the number of inner qubits and j is the number of steps moved in the orbit on the linear part. (each j removes 2 leaves)
    n = n + 1
    counter = 0
    pos = dict()
    theta = np.linspace(0, 2 * np.pi, n + 2 * j)
    theta = theta + (theta[1] / 2)
    for i in range(0, n + 2 * j - 1):
        if i < 2 * (2 * j):
            pos[i] = (np.cos(theta[(i)]), np.sin(theta[(i)]))
        else:
            pos[counter + i] = (2 * np.cos(theta[(i)]), 2 * np.sin(theta[(i)]))
            pos[counter + i + 1] = (np.cos(theta[(i)]), np.sin(theta[(i)]))
            counter += 1
    return pos


#
# s=nx.to_latex(res['g'][0], pos=pos_maker_rgs_orbit(9,0),node_label={i:"" for i in range(18)})
# print(s)

# %%
# nice figs
def nice_fig(X_set, Y_set, yerr=None, xerr=None, ax=None, fig=None, fit=False, box=True):
    """best used for scatter data"""
    # plot data
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6), dpi=200)
    ax.scatter(X_set, Y_set, edgecolors='b', s=50, alpha=0.85, label='sampled from orbit')
    ax.errorbar(X_set, Y_set, yerr=yerr, xerr=xerr)
    # Plot the best fit curve fit
    if fit:
        def func(x, A, B, C):
            return A - C * np.exp(B * ((-1) * x))

        popt, pcov = curve_fit(func, X_set[1:], Y_set[1:])
        X_fit = np.linspace(min(X_set), max(X_set), 500)
        Y_fit = func(X_fit, *popt)
        ax.plot(X_fit, Y_fit, linestyle='dashed', linewidth=2, alpha=0.5)
    # Set the title and labels
    ax.set_title('CNOT range coverage', fontsize=20)
    ax.set_xlabel('% of orbit checked', fontsize=16)
    ax.set_ylabel('% of CNOT range covered', fontsize=16)
    # Tweak the axis
    ax.set_xlim([0, np.floor(max(X_set)) + 1])
    ax.set_ylim([0.8 * min(Y_set) - 1, 1.2 * max(Y_set)])
    # Set grid
    ax.grid(False)
    # Set the font sizes
    ax.xaxis.label.set_size(14)
    ax.yaxis.label.set_size(14)
    ax.title.set_size(16)
    ax.xaxis.set_major_locator(ticker.AutoLocator())
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_major_locator(ticker.AutoLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.tick_params(axis='both', which='minor', labelsize=12)
    # Optional: remove top and right spines for a cleaner look
    ax.spines['top'].set_visible(box)
    ax.spines['right'].set_visible(box)
    plt.show()
    return ax, fig


# %%
def nice_fig_cor(X_set, Y_set, yerr=None, xerr=None, fit=False, ax=None, fig=None, box=True):
    """best used for linear correlation data"""

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6), dpi=200)
    ax.scatter(X_set, Y_set, edgecolors='b', s=30, alpha=0.85)
    if xerr or yerr:
        ax.errorbar(X_set, Y_set, yerr=yerr, xerr=xerr, fmt='o', capsize=3, ecolor='red', elinewidth=1)

    def func(x, A, B):
        return A * x + B

    if fit:
        popt, pcov = curve_fit(func, X_set, Y_set)
        X_fit = np.linspace(min(X_set), max(X_set), 500)
        Y_fit = func(X_fit, *popt)
        ax.plot(X_fit, Y_fit, linestyle='dashed', linewidth=2, alpha=0.5)
    ax.set_title('CNOT - Edge correlation', fontsize=20)
    ax.set_xlabel('Number of edges', fontsize=16)
    ax.set_ylabel('Average Number of CNOTs', fontsize=16)
    # Tweak the axis
    ax.set_xlim([0.9 * min(X_set), 1.1 * max(X_set)])
    ax.set_ylim([0.9 * min(Y_set), 1.1 * max(Y_set)])
    # ax.set_xticks(range(10, 23))
    # ax.set_yticks(range(8, 14))
    ax.xaxis.set_major_locator(ticker.AutoLocator())
    ax.yaxis.set_major_locator(ticker.AutoLocator())

    # Set grid
    ax.grid(False)
    # Set the font sizes
    ax.tick_params(axis='both', labelsize=12)
    ax.xaxis.label.set_size(14)
    ax.yaxis.label.set_size(14)
    ax.title.set_size(16)
    # Optional: remove top and right spines for a cleaner look
    ax.spines['top'].set_visible(box)
    ax.spines['right'].set_visible(box)
    plt.show()
    return ax, fig


def add_inset(X, Y, ax, x_lim=(135, 250)):
    ax_in = ax.inset_axes([0.15, 0.15, 0.3, 0.25])
    xin = []
    yin = []
    for i, x in enumerate(X):
        # limits of part of x-axis to show on the inset
        if x_lim[0] < x < x_lim[1]:
            xin.append(x)
            yin.append(Y[i])
    ax_in.scatter(xin, yin, edgecolors='b', s=22, alpha=0.85)
    ax_in.set_xlim(x_lim[0], x_lim[1])
    ax_in.set_ylim(0.95 * min(yin), 1.05 * max(yin))
    ax_in.tick_params(axis='both', labelsize=12)

    ax.indicate_inset_zoom(ax_in)

    return ax_in


# %%
from qiskit import QuantumCircuit


def circ2latex(circ):
    qc = QuantumCircuit.from_qasm_str(circ.to_openqasm())
    fig, ax = plt.subplots()
    style = {"displaytext": {}}
    ll = qc.draw(output="latex_source", ax=ax, plot_barriers=False, style=style)
    print(ll)


def edge_list2latex(edge_list, pos=None):
    g = nx.Graph()
    g.add_edges_from(edge_list)
    latex_code = nx.to_latex(g, pos=nx.circular_layout(g) if pos is None else pos)
    print(latex_code)


# s=nx.to_latex(g, pos=pos_maker_rgs_orbit(9,0),node_label={i:"" for i in range(18)})
# print(s)

def edgelist2graph(edge_list):
    g = nx.Graph()
    g.add_edges_from(edge_list)
    h = nx.Graph()
    h.add_nodes_from(sorted(g.nodes(data=True)))
    h.add_edges_from(g.edges(data=True))
    return h
