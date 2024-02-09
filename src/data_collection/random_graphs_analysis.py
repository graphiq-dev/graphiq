import time

import networkx as nx
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import linregress
from src.data_collection.user_interface import *
import os
import json
import itertools
from matplotlib.ticker import LogLocator, LogFormatter


# %%
# user_input = InputParams(
#     n_ordering=1,  # input
#     rel_inc_thresh=0.2,  # advanced: (0,1) The closer to 0 the closer we get to an exhaustive search for reordering.
#     allow_exhaustive=True,  # advanced*: only reason to deactivate is to save runtime if this is the bottleneck
#     iso_thresh=None,  # advanced: if not enough relabeled graphs are found, set it to a larger number!
#     n_lc_graphs=1,  # input
#     lc_orbit_depth=None,  # advanced: if hit the runtime limit, limit len(sequence of Local complementations)
#     lc_method=None,  # input
#     noise_simulation=False,  # input
#     noise_model_mapping="depolarizing",  # input
#     depolarizing_rate=0.005,  # input
#     error_margin=0.1,  # input
#     confidence=1,  # input
#     mc_map=None,  # advanced*: pass a manual noise map for the monte carlo simulations
#     n_cores=8,  # advanced: change if processor has different number of cores
#     seed=1,  # input
#     graph_type="linear",  # input
#     graph_size=2,  # input
#     verbose=False,
#     save_openqasm="none")


def crazy_list_maker(row, col):
    return [row for x in range(col)]


# %%

# graph = crazy(crazy_list_maker(2, 2))
# orbit = lc_orbit_finder(graph, comp_depth=None, orbit_size_thresh=None, with_iso=False, rand=False, rep_allowed=False)
# print(len(orbit))
times_list = []
orbit_sizes = []
for x in range(4, 4):
    graph = crazy(crazy_list_maker(3, x), alternate_order=False)
    s = time.time()
    orbit = lc_orbit_finder(graph, comp_depth=None, orbit_size_thresh=None, with_iso=False, rand=False,
                            rep_allowed=False)
    times_list = []
    s = time.time()
    e = time.time()
    times_list.append(e - s)
    orbit_sizes.append(len(orbit))
print("orbit growth", orbit_sizes, "\nrun time growth", times_list)


# %%
@ray.remote
def analyze_each_rnd_graph(n_nodes, p, indx, n_lc, graph_met, circ_met, swap):
    """to be used in rand_graph_sample_analysis"""
    g = nx.erdos_renyi_graph(n_nodes, p, seed=indx)
    s = time.time()
    orbit = lc_orbit_finder(g, comp_depth=None, orbit_size_thresh=n_lc, with_iso=True, rand=True,
                            rep_allowed=False)
    res3_list = []
    for g_i in orbit:
        res3_list.append(orbit_analyzer(g_i, dir_name='rnd_sampling', n_lc=1, circ_met_list=[circ_met],
                                        graph_met_list=[graph_met], plots=False))
    e = time.time()
    print(indx, f": {round(e - s, 2)} s; ", end='')
    circ_met_results = [res[circ_met][0] for res in res3_list]
    graph_met_results = [res['graph_metric'][0][graph_met] for res in res3_list]
    if swap:
        corr = corr_with_mean(graph_met_results, circ_met_results, print_result=False)
    else:
        corr = corr_with_mean(circ_met_results, graph_met_results, print_result=False)
    delta_t = e - s
    return corr, delta_t


def rand_graph_sample_analysis(n_samples, n_nodes=9, p=0.9, n_lc=None, graph_met='n_edges',
                               circ_met='max_emit_eff_depth', swap=False):
    correlations = []
    times_list = []
    print("samples analyzed:", end='')

    ray.shutdown()
    ray.init()

    futures = [analyze_each_rnd_graph.remote(n_nodes, p, indx, n_lc, graph_met, circ_met, swap) for indx
               in range(n_samples)]
    for future in ray.get(futures):
        corr, delta_t = future
        times_list.append(delta_t)
        correlations.append(corr)

    pearson_coeffs = [xx[0] for xx in correlations]
    avg_cor_data = np.average(pearson_coeffs), np.std(pearson_coeffs)
    print(f"\naverage correlation between {graph_met}, {circ_met} over {i} samples of random graphs of size {n_nodes} "
          f"with edge probability of {p} is {avg_cor_data[0]} +/- {avg_cor_data[1]}")
    print(f"average taken over {graph_met} for each value of {circ_met}")
    print(f"Overall runtime {round(sum(times_list), 2)}; average for each case: {round(np.average(times_list), 2)}; \n"
          f"Max: {round(max(times_list), 2)}; Min: {round(np.min(times_list), 2)}; Median: {round(np.median(times_list), 2)}")
    directory = f"/Users/sobhan/Desktop/EntgClass/Random cases/rand_sampling_attempt/nodes{n_nodes}_p0{round(p * 100)}_n_sample{n_samples}_{circ_met}"
    file_name_text = f"data_nodes{n_nodes}_p0{round(p * 100)}.txt"
    if not os.path.exists(directory):
        os.makedirs(directory)
    file_path = os.path.join(directory, file_name_text)
    with open(file_path, 'w') as f:
        for item in pearson_coeffs:
            f.write(f"{round(item, 3)}, ")
        f.write(
            f"\naverage correlation between {graph_met}, {circ_met} over {i} samples of random graphs of size {n_nodes}"
            f" with edge probability of {p} is {round(avg_cor_data[0], 3)} +/- {round(avg_cor_data[1], 3)}\n")
        f.write(
            f"Overall runtime {round(sum(times_list), 2)}; average for each case: {round(np.average(times_list), 2)}; \n"
            f"Max: {round(max(times_list), 2)}; Min: {round(np.min(times_list), 2)}; Median: {round(np.median(times_list), 2)}")
        f.write(f"\norbit size capped at {n_lc}")
    # for i in range(n_samples):
    #     file_name_res = f"res{i}"
    #     reses[i].save2json(directory, file_name_res)
    mean = np.mean(pearson_coeffs)
    std_dev = np.std(pearson_coeffs)

    # Plot histogram with 'auto' bins
    plt.hist(pearson_coeffs, bins='auto', alpha=0.7, color='blue', edgecolor='black')
    plt.title('Histogram of Data')
    plt.xlabel('Value')
    plt.ylabel('Frequency')

    # Show statistics
    plt.axvline(mean, color='red', linestyle='dashed', linewidth=1, label=f"Mean: {round(mean, 3)}")
    plt.axvline(mean - std_dev, color='green', linestyle='dashed', linewidth=1, label=f"1 Std Dev: {round(std_dev, 3)}")
    plt.axvline(mean + std_dev, color='green', linestyle='dashed', linewidth=1)
    plt.savefig(directory + "/pearson coeff distribution")
    plt.legend()
    plt.show()
    return correlations, avg_cor_data, times_list


@ray.remote
def analyze_each_sample(n_nodes, p, i, indx, n_lc, graph_met, smart_edge_reduction, extremum, circ_met_list):
    """to be used in rand_graph_cost_reduction"""
    print(indx, " ", end='')
    g = nx.erdos_renyi_graph(n_nodes, p, seed=int(i))
    if smart_edge_reduction:
        g_min = smart_edge_reducer(g, only_cluster=False)
    else:
        orbit = lc_orbit_finder(g, comp_depth=None, orbit_size_thresh=n_lc, with_iso=True, rand=True,
                                rep_allowed=True)
        graph_met_vals = [graph_met_value(graph_met, g) for g in orbit]
        g_min = orbit[graph_met_vals.index(extremum(graph_met_vals))]
    res1 = orbit_analyzer(g, dir_name='rnd_sampling', n_lc=1,
                          circ_met_list=circ_met_list, plots=False)
    res2 = orbit_analyzer(g_min, dir_name='rnd_sampling', n_lc=1,
                          circ_met_list=circ_met_list, plots=False)
    return res1, res2


def rand_graph_cost_reduction(n_samples, n_nodes=9, p=0.95, n_lc=100, graph_met='n_edges',
                              circ_met_list=['max_emit_eff_depth'], positive_cor=True, seed=99,
                              smart_edge_reduction=True):
    """

    :param n_samples: number of random graph to check the cost reduction for
    :param n_nodes: nodes of graph
    :param p: edge probability
    :param n_lc: size of the sample taken from orbit for each graph
    :param graph_met: graph metric that we want to check the reduction based up on
    :param circ_met: circuit cost metric
    :param positive_cor: true if the correlation is positive, false if it is negative
    :param smart_edge_reduction: only for n_edges as grpah metric, if true finds a LC graph with reduce edge by active
    method instead of random sampling
    :return: the list of tuples (initial cost, correlation reduced cost)
    """
    cost_tuples = {}
    reduction_percentages = {circ_met:[] for circ_met in circ_met_list}
    extremum = min if positive_cor else max
    rng = np.random.default_rng(seed)
    random_integers = rng.integers(0, int(1e6), size=n_samples)

    ray.shutdown()
    ray.init()

    futures = [analyze_each_sample.remote(n_nodes, p, indx, i, n_lc, graph_met, smart_edge_reduction, extremum,
                                          circ_met_list) for i, indx in enumerate(random_integers)]
    for future in ray.get(futures):
        res1, res2 = future
        for circ_met in circ_met_list:
            if circ_met in cost_tuples:
                cost_tuples[circ_met].append((res1[circ_met][0], res2[circ_met][0]))
            else:
                cost_tuples[circ_met] = [(res1[circ_met][0], res2[circ_met][0])]

    # Parallelized above
    # for indx, i in enumerate(random_integers):
    #     print(indx, " ", end='')
    #     g = nx.erdos_renyi_graph(n_nodes, p, seed=int(i))
    #     if smart_edge_reduction:
    #         g_min = smart_edge_reducer(g, only_cluster=False)
    #     else:
    #         orbit = lc_orbit_finder(g, comp_depth=None, orbit_size_thresh=n_lc, with_iso=True, rand=True,
    #                                 rep_allowed=True)
    #         graph_met_vals = [graph_met_value(graph_met, g) for g in orbit]
    #         g_min = orbit[graph_met_vals.index(extremum(graph_met_vals))]
    #     res1 = orbit_analyzer(g, dir_name='rnd_sampling', n_lc=1,
    #                           circ_met_list=circ_met_list, plots=False)
    #     res2 = orbit_analyzer(g_min, dir_name='rnd_sampling', n_lc=1,
    #                           circ_met_list=circ_met_list, plots=False)
    #     for circ_met in circ_met_list:
    #         if circ_met in cost_tuples:
    #             cost_tuples[circ_met].append((res1[circ_met][0], res2[circ_met][0]))
    #         else:
    #             cost_tuples[circ_met] = [(res1[circ_met][0], res2[circ_met][0])]
    for circ_met in circ_met_list:
        for x in cost_tuples[circ_met]:
            if x[0] != 0:
                reduction_percentages[circ_met].append((1 - x[1] / x[0]) * 100)
            else:
                reduction_percentages[circ_met].append(0)
        print(f"\navg {circ_met} reduction by {graph_met}:", round(np.mean(reduction_percentages[circ_met])), "% "
                                                                                                              "+/-",
              round(np.std(reduction_percentages[circ_met])), "%")
    # the output is cost_tuples which is a dictionary keyed by circuit metrics given.
    # for each metric, the value in dict is a list of tuples, the first item in tuple is the metric for the inital grpah
    # and the second item is the metric for the optimized graph. The optimization is done with respect to sth that is
    # correlated with number of edges since we look for a graph with reduced edge count.

    directory = f"/Users/sobhan/Desktop/EntgClass/Random cases/Sophias diagram results/{n_samples}samples_seed{seed}"
    file_name_costs = f"costs_nodes{n_nodes}_p0{round(p * 100)}.json"
    file_name_reductions = f"reductions_nodes{n_nodes}_p0{round(p * 100)}.json"
    file_name_txt = f"reductions_nodes{n_nodes}_p0{round(p * 100)}.txt"
    if not os.path.exists(directory):
        os.makedirs(directory)
    file_path1 = os.path.join(directory, file_name_costs)
    file_path2 = os.path.join(directory, file_name_reductions)
    file_path3 = os.path.join(directory, file_name_txt)
    with open(file_path1, 'w') as file:
        json.dump(cost_tuples, file)
    with open(file_path2, 'w') as file:
        json.dump(reduction_percentages, file)
    with open(file_path3, 'w') as f:
        for circ_met in circ_met_list:
            f.write(f"\navg {circ_met} reduction by {graph_met}: {round(np.mean(reduction_percentages[circ_met]))} % "
                    f"+/- {round(np.std(reduction_percentages[circ_met]))})% ")
        f.write(f"\n\ncost tuples\n{cost_tuples}"f"\n\nreduction %s\n{reduction_percentages}")
    return cost_tuples, reduction_percentages


# %%
def read_results(n_samples, seed, logx=True, logy=True, n_nodes_list=None, circ_met_list=None, met_names=None,
                 fig_size=(7.5, 6), dpi=400):
    p = 0.95
    cost_dict_list = []
    reduction_dict_list = []

    if circ_met_list is None:
        circ_met_list = ["n_cnots", "n_unitary", 'max_emit_reset_depth']
    if met_names is None:
        met_names = ["# CNOTs", "# Unitaries", "Emitter depth"]
    if n_nodes_list is None:
        n_nodes_list = [*range(15, 70, 5)]
    for n_nodes in n_nodes_list:
        directory = f"/Users/sobhan/Desktop/EntgClass/Random cases/Sophias diagram results/{n_samples}samples_seed{seed}"
        file_name_costs = f"costs_nodes{n_nodes}_p0{round(p * 100)}.json"
        file_name_reductions = f"reductions_nodes{n_nodes}_p0{round(p * 100)}.json"
        file_path1 = os.path.join(directory, file_name_costs)
        file_path2 = os.path.join(directory, file_name_reductions)
        with open(file_path1, 'r') as file:
            cost_dict_list.append(json.load(file))
        with open(file_path2, 'r') as file:
            reduction_dict_list.append(json.load(file))
    file_path3 = os.path.join(directory, f"list_of_cost_dicts_old_new{n_nodes}_p0{round(p * 100)}.txt")
    with open(file_path3, 'w') as f:
        f.write(f"\nFor each size, we have a dict keyed by metrics pointing to [old, new value]\n\n{cost_dict_list}")
    y_initial = {}
    y_final = {}
    y_initial_err = {}
    y_final_err = {}
    for i, n in enumerate(n_nodes_list):
        for met in circ_met_list:
            initial_costs = [x[0] for x in cost_dict_list[i][met]]
            final_costs = [x[1] for x in cost_dict_list[i][met]]
            if met in y_initial:
                y_initial[met].append(np.mean(initial_costs))
                y_final[met].append(np.mean(final_costs))
                y_initial_err[met].append(np.std(initial_costs))
                y_final_err[met].append(np.std(final_costs))
            else:
                y_initial[met] = [np.mean(initial_costs)]
                y_final[met] = [np.mean(final_costs)]
                y_initial_err[met] = [np.std(initial_costs)]
                y_final_err[met] = [np.std(final_costs)]
    # plot lines for each metric in circ_met_list
    x_values = n_nodes_list  # Common X-values for all lines
    y_values_i = [y_initial[met] for met in circ_met_list]
    errors_i = [y_initial_err[met] for met in circ_met_list]  # Errors for each line
    y_values_f = [y_final[met] for met in circ_met_list]
    errors_f = [y_final_err[met] for met in circ_met_list]  # Errors for each line
    # Different markers for each line
    markers = itertools.cycle(('o', 's', '^', 'D'))
    colors1 = itertools.cycle(['red', 'blue', 'cyan', 'orange',])
    colors2 = itertools.cycle([
        (1, 0.6, 0, 0.80),  # orangish red
        (0, 0.65, 1, 0.85),  # lightish blue
        (0, 0.35, 0.35),  # Dark cyan
        (0.5, 0.75, 0),  # Dark Orange
        # Add more colors if needed
    ])

    fig, ax = plt.subplots(figsize=fig_size, dpi=dpi)
    for y, err, marker, label, color in zip(y_values_i, errors_i, markers, met_names, colors1):
        ax.errorbar(x_values, y, yerr=err, fmt=marker + '-', color=color, capsize=6, label=f'{label}')
    for y, err, marker, label, color in zip(y_values_f, errors_f, markers, met_names, colors2):
        ax.errorbar(x_values, y, yerr=err, fmt=marker + '--', color=color, capsize=6, label=f'Optimized: {label}')

    if logx:
        ax.set_xscale('log')
        ax.xaxis.set_major_locator(plt.NullLocator())
        ax.xaxis.set_minor_locator(plt.NullLocator())
        ax.set_xticks(n_nodes_list)
        ax.set_xticklabels(n_nodes_list)
    if logy:
        ax.set_yscale('log')
        major_locator = LogLocator(base=10.0)
        minor_locator = LogLocator(base=10.0, subs=[2, 3, 4, 5, 6, 7, 8, 9])
        ax.yaxis.set_major_locator(major_locator)
        ax.yaxis.set_minor_locator(minor_locator)
        # formatter = LogFormatter(base=10.0, labelOnlyBase=False, minor_thresholds=(100, 0.001))
        # ax.yaxis.set_major_formatter(formatter)
        # # Optionally, if you want labels on minor ticks as well
        # ax.yaxis.set_minor_formatter(formatter)

    # Adding legend
    plt.legend()
    plt.legend(frameon=False, fontsize=12)

    # Adding title and labels
    # Set the font sizes
    ax.tick_params(axis='both', labelsize=12)

    # plt.title('Metric Scaling')
    plt.xlabel('Number of Photons', fontsize=14)
    plt.ylabel('Circuit Metrics', fontsize=14)

    # Show plot
    plt.show()
    return cost_dict_list


# %% edge reduction block
def smart_edge_reducer(g, only_cluster=False, deg_exp=1, normalized_deg=False, show=False):
    g_new = nx.Graph(g)
    g_new2 = nx.Graph(g)
    n_edges_i = g.number_of_edges()
    n_edges = 0
    counter = 0
    pos = nx.circular_layout(g)
    while n_edges < n_edges_i:
        if counter > 0:
            n_edges_i = n_edges
        counter += 1
        g_new = g_new2
        clust_dict = nx.clustering(g_new)
        deg_dict = dict(g_new.degree)
        if normalized_deg:
            for k in deg_dict:
                deg_dict[k] = deg_dict[k] / (len(g) - 1)
        subgraph_edges_dict = dict(zip([n for n in clust_dict], [clust_dict[n] * (deg_dict[n] ** deg_exp)
                                                                 for n in clust_dict]))
        if only_cluster:
            node = max(clust_dict, key=clust_dict.get)
        else:
            node = max(subgraph_edges_dict, key=subgraph_edges_dict.get)
        if show:
            nx.draw_networkx(g_new, pos=pos)
            plt.show()
        g_new2 = local_comp_graph(g_new, node)
        n_edges = g_new2.number_of_edges()
    # print("edge reduction from", g.number_of_edges(), "to", g_new.number_of_edges(), "over", counter
    #       , "steps")
    return g_new


def edge_strategy_tester(n_samples, n_nodes=20, p=0.95, seed=999):
    n_edges_original = []
    n_edges_cluster = []
    n_edges_cluster_deg = []
    n_edges_cluster_deg_norm = []
    n_edges_cluster_deg2 = []
    n_edges_cluster_deg2_norm = []
    rng = np.random.default_rng(seed)
    random_integers = rng.integers(0, int(1e6), size=n_samples)
    for indx, i in enumerate(random_integers):
        print(indx, " ", end='')
        g = nx.erdos_renyi_graph(n_nodes, p, seed=int(i))
        n_edges_original.append(g.number_of_edges())
        n_edges_cluster.append(
            smart_edge_reducer(g, only_cluster=True, deg_exp=1, normalized_deg=False).number_of_edges())
        n_edges_cluster_deg.append(
            smart_edge_reducer(g, only_cluster=False, deg_exp=1, normalized_deg=False).number_of_edges())
        n_edges_cluster_deg_norm.append(
            smart_edge_reducer(g, only_cluster=False, deg_exp=1, normalized_deg=True).number_of_edges())
        n_edges_cluster_deg2.append(
            smart_edge_reducer(g, only_cluster=False, deg_exp=2, normalized_deg=False).number_of_edges())
        n_edges_cluster_deg2_norm.append(
            smart_edge_reducer(g, only_cluster=False, deg_exp=2, normalized_deg=True).number_of_edges())

    strategies = [n_edges_cluster, n_edges_cluster_deg, n_edges_cluster_deg_norm, n_edges_cluster_deg2,
                  n_edges_cluster_deg2_norm]
    max_red_possible = n_edges_original[:]
    for i in range(n_samples):
        max_red_possible[i] = (n_edges_original[i] - (n_nodes - 1)) / n_edges_original[i] * 100
        for strategy in strategies:
            strategy[i] = (n_edges_original[i] - strategy[i]) / n_edges_original[i] * 100
    for j, strategy in enumerate(strategies):
        strategies[j] = round(np.mean(strategy), 2), round(np.std(strategy), 2)
    max_red_possible = round(np.mean(max_red_possible), 2), round(np.std(max_red_possible), 2)
    print(f"\nstrategies:\n n_edges_cluster: {strategies[0]},\n n_edges_cluster_deg: {strategies[1]},\n "
          f"n_edges_cluster_deg_norm: {strategies[2]},\n n_edges_cluster_deg2: {strategies[3]},"
          f"\n n_edges_cluster_deg2_norm: {strategies[4]}")
    print("\ndifference to max reduction possible in the same order:",
          [(round((strategy[0] - max_red_possible[0]), 2),
            round(np.sqrt(np.square(strategy[1]) + np.square(max_red_possible[1])), 2)) for strategy in strategies])
    return strategies, max_red_possible


def sampling_vs_smart(n_trials, n_lc, n_nodes=20, p=0.95, seed=999):
    rng = np.random.default_rng(seed)
    random_integers = rng.integers(0, int(1e6), size=n_trials)
    reduction_diffs = []
    for indx, i in enumerate(random_integers):
        print(indx, " ", end='')
        g = nx.erdos_renyi_graph(n_nodes, p, seed=int(i))
        g_min_smart = smart_edge_reducer(g)
        orbit = lc_orbit_finder(g, comp_depth=None, orbit_size_thresh=n_lc, with_iso=True, rand=True,
                                rep_allowed=False)
        graph_met_vals = [g.number_of_edges() for g in orbit]
        g_min_sample = orbit[graph_met_vals.index(min(graph_met_vals))]
        x0 = g.number_of_edges()
        x1 = g_min_smart.number_of_edges()
        x2 = g_min_sample.number_of_edges()
        reduction1 = (x0 - x1) / x0 * 100
        reduction2 = (x0 - x2) / x0 * 100
        delta_reduction = reduction1 - reduction2
        reduction_diffs.append(delta_reduction)
    print(f"\n% reduction difference between active edge reduction vs the sampling with {n_lc} "
          f"random cases of the LC orbit studies over {n_trials} trials")
    print(f"{round(np.mean(reduction_diffs), 2)} ± {round(np.std(reduction_diffs), 2)}")


# %%
@ray.remote
def parallel_ops(indx, random_integers, n_lc_samples, n_nodes, p, strategies, circ_met_list=[]):
    active_runtime, rnd_runtime, solve_runtime = 0, 0, 0
    i = random_integers[indx]
    print(indx, " ", end='')
    g = nx.erdos_renyi_graph(n_nodes, p, seed=int(i))
    res0 = orbit_analyzer(g, dir_name='rnd_sampling', n_lc=1, circ_met_list=circ_met_list,
                          graph_met_list=[], plots=False)
    if "active" in strategies:
        t1 = time.time()
        g_min = smart_edge_reducer(g, only_cluster=False)
        res1 = orbit_analyzer(g_min, dir_name='rnd_sampling', n_lc=1, circ_met_list=circ_met_list,
                              graph_met_list=[], plots=False)
        t2 = time.time()
        active_runtime = t2 - t1
    if "rnd" in strategies:
        t1 = time.time()
        orbit = lc_orbit_finder(g, comp_depth=None, orbit_size_thresh=n_lc_samples, with_iso=True, rand=True,
                                rep_allowed=False)
        graph_met_vals = [g.number_of_edges() for g in orbit]
        g_min = orbit[graph_met_vals.index(min(graph_met_vals))]
        t2 = time.time()
        rnd_runtime = t2 - t1
        res2 = orbit_analyzer(g_min, dir_name='rnd_sampling', n_lc=1, circ_met_list=circ_met_list,
                              graph_met_list=[], plots=False)
        if "solve" in strategies:
            res3_list = []
            for g_i in orbit:
                res3_list.append(orbit_analyzer(g_i, dir_name='rnd_sampling', n_lc=1, circ_met_list=circ_met_list,
                                                graph_met_list=[], plots=False))
            res3_list.sort(key=lambda res: min(res[circ_met_list[0]]))
            res3 = res3_list[0]
            t2 = time.time()
            solve_runtime = t2 - t1
        # best circuit metric results for active, random, and solving the full sample
        best_circ_met_results = res0[circ_met_list[0]], res1[circ_met_list[0]], res2[circ_met_list[0]], res3[
            circ_met_list[0]]
        return active_runtime, rnd_runtime, solve_runtime, best_circ_met_results


def runtime_analysis(n_samples, n_lc_samples, n_nodes, p, strategies=None, seed=99, circ_met_list=[]):
    if strategies is None:
        strategies = ["rnd", "active", "solve"]
    rng = np.random.default_rng(seed)
    random_integers = rng.integers(0, int(1e6), size=n_samples)
    active_runtimes = []
    rnd_runtimes = []
    solve_runtimes = []

    circ_met_initial = []
    circ_met_active = []
    circ_met_rnd = []
    circ_met_solve = []

    ray.shutdown()
    ray.init()

    futures = [parallel_ops.remote(indx, random_integers, n_lc_samples, n_nodes, p, strategies, circ_met_list) for indx
               in range(n_samples)]
    for future in ray.get(futures):
        active_runtime, rnd_runtime, solve_runtime, circ_met_result = future
        active_runtimes.append(active_runtime)
        rnd_runtimes.append(rnd_runtime)
        solve_runtimes.append(solve_runtime)
        circ_met_initial.append(circ_met_result[0])
        circ_met_active.append(circ_met_result[1])
        circ_met_rnd.append(circ_met_result[2])
        circ_met_solve.append(circ_met_result[3])

    ray.shutdown()

    # for indx, i in enumerate(random_integers):
    #     print(indx, " ", end='')
    #     g = nx.erdos_renyi_graph(n_nodes, p, seed=int(i))
    #     if "active" in strategies:
    #         t1 = time.time()
    #         g_min = smart_edge_reducer(g, only_cluster=False)
    #         t2 = time.time()
    #         active_runtime = t2 - t1
    #         active_runtimes.append(t2 - t1)
    #     if "rnd" in strategies:
    #         t1 = time.time()
    #         orbit = lc_orbit_finder(g, comp_depth=None, orbit_size_thresh=n_lc_samples, with_iso=True, rand=True,
    #                                 rep_allowed=False)
    #         graph_met_vals = [graph_met_value("n_edges", g) for g in orbit]
    #         g_min = orbit[graph_met_vals.index(min(graph_met_vals))]
    #         t2 = time.time()
    #         rnd_runtime = t2 - t1
    #         rnd_runtimes.append(t2 - t1)
    #         for g_i in orbit:
    #             res2 = orbit_analyzer(g_i, dir_name='rnd_sampling', n_lc=1, circ_met_list=[], graph_met_list=[],
    #                                   plots=False)
    #         t2 = time.time()
    #         solve_runtime = t2 - t1
    #         solve_runtimes.append(t2 - t1)
    directory = f"/Users/sobhan/Desktop/EntgClass/Random cases/runtimes/{n_samples}samples_seed{seed}"
    file_name_txt = f"n_lc{n_lc_samples}_n_nodes{n_nodes}_p0{round(p * 100)}.txt"
    if not os.path.exists(directory):
        os.makedirs(directory)
    file_path = os.path.join(directory, file_name_txt)
    with open(file_path, 'w') as f:
        f.write(f"active_runtimes {round(np.mean(active_runtimes), 6)} ± {round(np.std(active_runtimes), 6)}")
        f.write(f"\nrand_runtimes {round(np.mean(rnd_runtimes), 6)} ± {round(np.std(rnd_runtimes), 6)}")
        f.write(f"\nsolve_runtimes {round(np.mean(solve_runtimes), 6)} ± {round(np.std(solve_runtimes), 6)}")
        f.write(
            f"\noriginal_{circ_met_list[0]}: {round(np.mean(circ_met_initial), 6)} ± {round(np.std(circ_met_initial), 6)}")
        f.write(
            f"\nactive_{circ_met_list[0]}: {round(np.mean(circ_met_active), 6)} ± {round(np.std(circ_met_active), 6)}")
        f.write(f"\nrand_{circ_met_list[0]}: {round(np.mean(circ_met_rnd), 6)} ± {round(np.std(circ_met_rnd), 6)}")
        f.write(f"\nsolve_{circ_met_list[0]}: {round(np.mean(circ_met_solve), 6)} ± {round(np.std(circ_met_solve), 6)}")
        f.write(f"\nthe circuit metris is {circ_met_list[0]}")
        f.write(f"\n\n{active_runtimes}\n\n{rnd_runtimes}\n\n{solve_runtimes}\n")
        f.write(f"\n\n{circ_met_active}\n\n{circ_met_initial}\n\n{circ_met_rnd}\n\n{circ_met_solve}")
    return active_runtimes, rnd_runtimes, solve_runtimes, circ_met_initial, circ_met_active, circ_met_rnd, circ_met_solve


# %% fitting
def power_law(x, a, b):
    return a * np.power(x, b)


def exponential(x, a, b, c):
    x = np.array(x)
    return a - b * np.exp(-c * x)


def fit_runtimes(n_nodes, active, active_err, rnd, rnd_err, solve, solve_err, logx=True, logy=True):
    if not active:
        active = [0.003452, 0.004758, 0.006213, 0.007634, 0.009773, 0.010263, 0.010978, 0.023758]
        active_err = [0.001447, 0.002077, 0.003016, 0.003232, 0.007085, 0.00715, 0.007003, 0.004525]
        rnd = [0.435869, 0.5285, 0.619553, 0.725905, 0.812355, 0.930152, 1.060124, 3.313189]
        rnd_err = [0.109694, 0.09301, 0.076958, 0.09788, 0.11461, 0.158822, 0.177062, 0.543116]
        solve = [15.497743, 22.135777, 30.166978, 40.012873, 50.186908, 63.26804, 80.131508]
        solve_err = [3.43016, 3.170175, 3.178408, 4.663032, 5.366486, 6.849141, 8.708934]
        # n0 = [393.44, 342.23, 295.94, 256.86, 215.12, 184.31, 145.85]
        # n0err = [45.031393, 44.495585, 33.599649, 31.281311, 23.34921, 21.095353, 15.764755]
        n_nodes = [*range(15, 36, 3)] + [60]

    x_data = np.array(n_nodes)
    # y0_data = np.array(n0) * 1e0
    # errors0 = np.array(n0err) * 1e0
    y1_data = np.array(active) * 1e3
    errors1 = np.array(active_err) * 1e3
    y2_data = np.array(rnd) * 1e3
    errors2 = np.array(rnd_err) * 1e3
    # y3_data = np.array(solve) * 1e3
    # errors3 = np.array(solve_err) * 1e3

    # Perform the curve fitting
    # params0, covariance0 = curve_fit(power_law, x_data, y0_data, sigma=errors0, absolute_sigma=True)
    params1, covariance1 = curve_fit(power_law, x_data, y1_data, sigma=errors1, absolute_sigma=True)
    params2, covariance2 = curve_fit(power_law, x_data, y2_data, sigma=errors2, absolute_sigma=True)
    # params3, covariance3 = curve_fit(power_law, x_data, y3_data, sigma=errors3, absolute_sigma=True)

    # Extracting the parameters
    # a0, b0 = params0
    a1, b1 = params1
    a2, b2 = params2
    # a3, b3 = params3

    # Generate fitted data for plotting
    # fitted_y0 = power_law(x_data, a0, b0)
    fitted_y1 = power_law(x_data, a1, b1)
    fitted_y2 = power_law(x_data, a2, b2)
    # fitted_y3 = power_law(x_data, a3, b3)

    # Plotting

    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    if logy:
        ax.set_yscale('log')
        # major_locator = LogLocator(base=10.0)
        # minor_locator = LogLocator(base=10.0, subs=[2, 3, 4, 5, 6, 7, 8, 9])
        # ax.yaxis.set_major_locator(major_locator)
        # ax.yaxis.set_minor_locator(minor_locator)
        # formatter = LogFormatter(base=10.0, labelOnlyBase=False, minor_thresholds=(100, 0.001))
        # ax.yaxis.set_major_formatter(formatter)
        # # Optionally, if you want labels on minor ticks as well
        # ax.yaxis.set_minor_formatter(formatter)
    if logx:
        ax.set_xscale('log')
        ax.xaxis.set_major_locator(plt.NullLocator())
        ax.xaxis.set_minor_locator(plt.NullLocator())
        ax.set_xticks(n_nodes)
        ax.set_xticklabels(n_nodes)
    # plt.errorbar(x_data, y0_data, yerr=errors0, fmt='o', color='black', capsize=5)
    # plt.plot(x_data, fitted_y0, label=f'Initial: t = {a0:.2f}n^{b0:.2f}', color='black')
    plt.errorbar(x_data, y1_data, yerr=errors1, fmt='o', color='red', capsize=5)
    plt.plot(x_data, fitted_y1, label=f'Active: t = {a1:.2f}n^{b1:.2f}', color='red')
    plt.errorbar(x_data, y2_data, yerr=errors2, fmt='o', color='blue', capsize=5)
    plt.plot(x_data, fitted_y2, label=f'Sampling: t = {a2:.2f}n^{b2:.2f}', color='blue')
    # plt.errorbar(x_data, y3_data, yerr=errors3, fmt='o', color='green', capsize=5)
    # plt.plot(x_data, fitted_y3, label=f'Solving: t = {a3:.2f}n^{b3:.2f}', color='green')
    plt.legend()
    plt.xlabel('Number of Photons')
    plt.ylabel('Runtime (ms)')
    plt.title('Runtime scaling of different methods')
    plt.show()


def fit_sample_vs_smart(x_data, y_data, y_error, fit_func=power_law):
    params1, covariance1 = curve_fit(fit_func, x_data, y_data, p0=np.array([5, 7, 0.014]), sigma=y_error,
                                     absolute_sigma=True)
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    # ax.set_yscale('log')
    a1, b1, c1 = params1
    fitted_y1 = fit_func(x_data, a1, b1, c1)
    plt.errorbar(x_data, y_data, yerr=y_error, fmt='o', color=(0, 0, 0.6), capsize=5)
    plt.plot(x_data, fitted_y1, label=f': %diff = {a1:.2f}+{abs(b1):.2f} exp(-{c1:.3f} N)', color='blue')
    plt.legend()
    plt.xlabel('Orbit Sample Size')
    plt.ylabel('% edge reduction difference')
    plt.title('Edge reduction difference: Active vs Sampling methods\nn=30, p=0.95, over 100 trials')
    plt.show()
    return params1, covariance1

# %%
# To find the average cost reduction on average over some number of graph over a sample of their orbit
# rand_graph_cost_reduction(100, n_nodes=18, p=0.95, n_lc=100, graph_met='n_edges', circ_met='max_emit_eff_depth',
#                               positive_cor=True, seed=88)
# edge_strategy_tester(1000, 15, p=0.80,seed=21)
# res = rand_graph_cost_reduction(200, n_nodes=18, p=0.95, seed=9, circ_met='max_emit_depth', smart_edge_reduction=True)
