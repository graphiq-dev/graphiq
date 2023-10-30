import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import linregress
from src.data_collection.user_interface import *
import os


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

graph = crazy(crazy_list_maker(2, 2))
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


def rand_graph_sample_analysis(n_samples, n_nodes=9, p=0.9, n_lc=None, graph_met='n_edges',
                               circ_met='max_emit_eff_depth'):
    reses = []
    correlations = []
    times_list = []
    print("samples analyzed:", end='')
    for i in range(n_samples):
        g = nx.erdos_renyi_graph(n_nodes, p, seed=i)
        s = time.time()
        res = orbit_analyzer(g, dir_name='rnd_sampling', n_lc=n_lc,
                             graph_met_list=[graph_met], plots=False)
        e = time.time()
        times_list.append(e - s)
        print(i, f": {round(e - s, 2)} s; ", end='')
        reses.append(res)
        x = corr_with_mean(res[circ_met], [res['graph_metric'][i][graph_met] for i in range(len(res))],
                           print_result=False)
        correlations.append(x)
    pearson_coeffs = [xx[0] for xx in correlations]
    avg_cor_data = np.average(pearson_coeffs), np.std(pearson_coeffs)
    print(f"\naverage correlation between {graph_met}, {circ_met} over {i} samples of random graphs of size {n_nodes} "
          f"with edge probability of {p} is {avg_cor_data[0]} +/- {avg_cor_data[1]}")
    print(f"average taken over {graph_met} for each value of {circ_met}")
    print(f"Overall runtime {round(sum(times_list), 2)}; average for each case: {round(np.average(times_list), 2)}; \n"
          f"Max: {round(max(times_list), 2)}; Min: {round(np.min(times_list), 2)}; Median: {round(np.median(times_list), 2)}")
    directory = f"/Users/sobhan/Desktop/EntgClass/Random cases/rand_sampling_attempt/nodes{n_nodes}_p0{round(p * 100)}"
    file_name_text = f"data_nodes{n_nodes}_p0{round(p * 100)}.txt"
    if not os.path.exists(directory):
        os.makedirs(directory)
    file_path = os.path.join(directory, file_name_text)
    with open(file_path, 'w') as f:
        for item in pearson_coeffs:
            f.write(f"{round(item, 3)}, ")
        f.write(
            f"\naverage correlation between {graph_met}, {circ_met} over {i} samples of random graphs of size {n_nodes}"
            f" with edge probability of {p} is {round(avg_cor_data[0],3)} +/- {round(avg_cor_data[1], 3)}\n")
        f.write(
            f"Overall runtime {round(sum(times_list), 2)}; average for each case: {round(np.average(times_list), 2)}; \n"
            f"Max: {round(max(times_list), 2)}; Min: {round(np.min(times_list), 2)}; Median: {round(np.median(times_list), 2)}")
        f.write(f"\norbit size capped at {n_lc}")
    for i in range(100):
        file_name_res = f"res{i}"
        reses[i].save2json(directory, file_name_res)
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
    plt.savefig(directory+"/pearson coeff distribution")
    plt.legend()
    plt.show()
    return reses, correlations, avg_cor_data, times_list


def rand_graph_cost_reduction(n_samples, n_nodes=9, p=0.95, n_lc=100, graph_met='n_edges',
                               circ_met='max_emit_eff_depth', positive_cor=True, seed=99):
    """

    :param n_samples: number of random graph to check the cost reduction for
    :param n_nodes: nodes of graph
    :param p: edge probability
    :param n_lc: size of the sample taken from orbit for each graph
    :param graph_met: graph metric that we want to check the reduction based up on
    :param circ_met: circuit cost metric
    :param positive_cor: true if the correlation is positive, false if it is negative
    :return: the list of tuples (initial cost, correlation reduced cost)
    """
    cost_tuples = []
    extermum = min if positive_cor else max
    rng = np.random.default_rng(seed)
    random_integers = rng.integers(0, int(1e6), size=n_samples)
    for indx, i in enumerate(random_integers):
        print(indx, " ", end='')
        g = nx.erdos_renyi_graph(n_nodes, p, seed=int(i))
        orbit = lc_orbit_finder(g, comp_depth=None, orbit_size_thresh=n_lc, with_iso=True, rand=True, rep_allowed=True)
        graph_met_vals = [graph_met_value(graph_met, g) for g in orbit]
        g_min = orbit[graph_met_vals.index(extermum(graph_met_vals))]
        res1 = orbit_analyzer(g, dir_name='rnd_sampling', n_lc=1,
                              graph_met_list=[graph_met], plots=False)
        res2 = orbit_analyzer(g_min, dir_name='rnd_sampling', n_lc=1,
                              graph_met_list=[graph_met], plots=False)
        cost_tuples.append((res1[circ_met][0], res2[circ_met][0]))
    reduction_percentages = [(1-x[1]/x[0])*100 for x in cost_tuples]
    print(f"\navg {circ_met} reduction by {graph_met}:", round(np.mean(reduction_percentages)), "% "
          "+/-", round(np.std(reduction_percentages)), "%")
    return cost_tuples, reduction_percentages

# %% rand_graph_cost_reduction(100, n_nodes=18, p=0.95, n_lc=100, graph_met='n_edges', circ_met='max_emit_eff_depth',
#                               positive_cor=True, seed=88)