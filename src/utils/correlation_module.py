from src.utils.relabel_module import *
import networkx as nx
import numpy as np
from benchmarks.graph_states import repeater_graph_states, star_graph_state, lattice_cluster_state
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from math import ceil
from src.backends.stabilizer.functions.height import height_max


# %%
def random_connected_graph(n, p, seed=None, directed=None):
    rnd_maker = nx.gnp_random_graph
    g = nx.Graph()
    g.add_edges_from(((0, 1), (2, 3)))
    while not nx.is_connected(g):
        g = rnd_maker(n, p, seed=seed, directed=directed)
    return g


# %% initialize
m = 16
# graph = lattice_cluster_state((3, 4)).data
# graph = star_graph_state(m).data
# graph = repeater_graph_states(m)
# graph = nx.random_tree(m)
graph = random_connected_graph(m, 0.95, seed=None, directed=False)
g1 = (nx.to_numpy_array(graph)).astype(int)
g2 = nx.from_numpy_array(g1)
g3 = nx.convert_node_labels_to_integers(g2)
adj = nx.to_numpy_array(g3)
n = len(g3)


# %% Utils
def adj_emit_list(adj, count=1000):
    # makes a number of isomorphic graphs out of the input one and returns both the adj list and #emitters list both
    # are sorted and there is one to one correspondence between the two lists.
    iso_count = min(np.math.factorial(n), count)
    arr = iso_finder(adj, iso_count, allow_exhaustive=True)
    li11 = emitter_sorted(arr)
    emit_list = [x[1] for x in li11]
    adj_list = [x[0] for x in li11]
    return adj_list, emit_list


def avg_maker(x_data, y_data):
    # for all similar items in data1 find the average of corresponding values in data2.
    avg_dict = {}  # key is item in data1 and value is tuple (avg_so_far, count_of_items) from data2
    x_data_uniq = []
    y_data_avg = []
    assert len(x_data) == len(y_data)
    for i in range(len(x_data)):
        x_value = x_data[i]
        if x_value in avg_dict:
            count = avg_dict[x_value][1]
            avg_so_far = avg_dict[x_value][0]
            # update
            new_avg = (count * avg_so_far + y_data[i]) / (count + 1)
            count += 1
            avg_dict[x_value] = (new_avg, count)
        else:
            avg_dict[x_value] = (y_data[i], 1)
    for item in sorted(avg_dict.items()):
        x_data_uniq.append(item[0])
        y_data_avg.append(item[1][0])
    return x_data_uniq, y_data_avg


# %% node based
# prepare data
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
    print('correlation:', overall_corr)
    # retry with average values over all cases with the same number of emitters
    emits_no_repeat, avg_corrs = avg_maker(emit_list, pear_corr_list)
    overall_avg_corr, _ = pearsonr(emits_no_repeat, avg_corrs)
    print('avg correlation:', overall_avg_corr)
    plt.plot(emits_no_repeat, avg_corrs)
    plt.show()
    return emits_no_repeat, avg_corrs


# %%
def num_emit_vs_p(n, trials=1000, p_step=0.1):
    num_emit_list = []
    avg_emit = []
    avg_emit_std = []
    p_list = [x * p_step for x in range(ceil(0.1 / p_step), int(1 / p_step))]
    for p in p_list:
        for i in range(trials):
            g = random_connected_graph(n, p)
            n_emit = height_max(graph=g)
            num_emit_list.append(n_emit)

        avg_emit.append(np.mean(num_emit_list))
        avg_emit_std.append(np.std(num_emit_list))
        num_emit_list = []
    fig = plt.figure(figsize=(7.5, 5.5), dpi=300)
    fig.tight_layout()
    plt.scatter(p_list, avg_emit)
    plt.errorbar(p_list, avg_emit, yerr=list(avg_emit_std), fmt='bo', ecolor='r', capsize=4, errorevery=4, capthick=2)
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
        g = random_connected_graph(n, p)
        n_emit = height_max(graph=g)
        num_emit_list.append(n_emit)
    num_emit_set = set(num_emit_list)
    count_percent_list = [(num_emit_list.count(x)/trials)*100 for x in num_emit_set]
    dist_dict = dict(zip(num_emit_set, count_percent_list))  # {num_emit: its count}
    avg = np.mean(num_emit_list)
    std = np.std(num_emit_list)
    print("mean and standard deviation:", avg, ",", std)
    if show_plot:
        fig = plt.figure(figsize=(8, 6), dpi=300)
        fig.tight_layout()
        plt.scatter(dist_dict.keys(), dist_dict.values())
        plt.figtext(0.75, 0.70, f"n: {n}\np: {p}\navg: {round(avg,2)}\nstd: {round(std,2)}\ntrials: "
                                f"{'{:.0e}'.format(trials)}")
        plt.title("Number of emitters for random graphs")
        plt.xlabel("Number of emitters")
        plt.ylabel("% percentage")
        plt.xticks(range(min(num_emit_set), max(num_emit_set)+1))
        plt.ylim(0, max(count_percent_list)*1.25)
        plt.show()
    return avg, std



# %% graph based: max betweenness
def _max_bet_min_emit(adj, trials=1000):
    # returns max betweenness and min number of emitters among the limited number of relabeling trials
    g = nx.from_numpy_array(adj)
    _, emit_list = adj_emit_list(adj, count=trials)
    dict_centrality = nx.betweenness_centrality(g)
    max_bet = max(dict_centrality.values())
    min_emit = emit_list[0]
    return max_bet, min_emit


def max_bet_min_emit_corr(n, p, n_samples=100, relabel_trials=100, show_plot=False):
    max_bet_list = []
    min_emit_list = []
    for i in range(n_samples):
        g = random_connected_graph(n, p)
        adj = nx.to_numpy_array(g)
        max_bet, min_emit = _max_bet_min_emit(adj, trials=relabel_trials)
        max_bet_list.append(max_bet)
        min_emit_list.append(min_emit)
    corr, _ = pearsonr(min_emit_list, max_bet_list)
    # average case
    n_emit, avg_bet = avg_maker(min_emit_list, max_bet_list)
    avg_corr, _ = pearsonr(n_emit, avg_bet)
    if show_plot:
        fig, (ax1, ax2) = plt.subplots(2, figsize=(15, 25))
        ax1.scatter(min_emit_list, max_bet_list)
        ax2.plot(n_emit, avg_bet)
        plt.show()
    print("correlation=", corr, "avg correlation=", avg_corr)
    return corr, avg_corr, (n_emit, avg_bet)


def p_of_edge_dependence(n, n_samples=1000, relabel_trials=2):
    # the max betweenness vs emitter number for different p values
    for p in range(1, 10):
        _, _, plot_pair = max_bet_min_emit_corr(n, p / 10, n_samples=n_samples, relabel_trials=relabel_trials)
        plt.plot(plot_pair[0], plot_pair[1], label=f'p={p / 10}')
    plt.legend(loc='best')
    plt.yscale("log")
    plt.show()


# %%
between_corr(adj, count=1000)

# %% whole graph based
p_of_edge_dependence(24, n_samples=500, relabel_trials=4)
# %%
max_bet_min_emit_corr(n, 0.9, n_samples=100, relabel_trials=100)

# %% visual
# fig, (ax1, ax2) = plt.subplots(2, figsize=(15, 25))
# fig.tight_layout()
# nx.draw_networkx(g3, with_labels=1, ax=ax1, pos=nx.kamada_kawai_layout(g3))
# nx.draw_networkx(new_g, with_labels=1, ax=ax2, pos=nx.kamada_kawai_layout(new_g))
# plt.show()

# %%
