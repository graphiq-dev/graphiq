###
# data collection script #
###
import time
import ray
import numpy as np
from math import factorial
import pandas as pd
from src.data_collection.ui_functions import *
from correlation_module import *
from scipy.stats import spearmanr, pearsonr

from src.utils.relabel_module import lc_orbit_finder

# %% initialize target graph
# options are "rnd": random, "tree": random tree, "rgs": repeater graph state, "linear": linear cluster state,
# "lattice": 2-d cluster state, "star": star graph or GHZ
# g = t_graph(gtype="rgs", n=4, seed=99, show=True)

# %%
adj1 = np.array([[0., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                 [1., 0., 1., 1., 1., 1., 1., 1., 1., 1.],
                 [1., 1., 0., 1., 0., 0., 0., 0., 0., 0.],
                 [1., 1., 1., 0., 0., 0., 0., 0., 0., 0.],
                 [1., 1., 0., 0., 0., 1., 0., 0., 0., 0.],
                 [1., 1., 0., 0., 1., 0., 0., 0., 0., 0.],
                 [1., 1., 0., 0., 0., 0., 0., 1., 0., 0.],
                 [1., 1., 0., 0., 0., 0., 1., 0., 0., 0.],
                 [1., 1., 0., 0., 0., 0., 0., 0., 0., 1.],
                 [1., 1., 0., 0., 0., 0., 0., 0., 1., 0.]])
user_input = InputParams(
    n_ordering=1,  # input
    rel_inc_thresh=0.2,  # advanced: (0,1) The closer to 0 the closer we get to an exhaustive search for reordering.
    allow_exhaustive=True,  # advanced*: only reason to deactivate is to save runtime if this is the bottleneck
    iso_thresh=None,  # advanced: if not enough relabeled graphs are found, set it to a larger number!
    n_lc_graphs=1,  # input
    lc_orbit_depth=None,  # advanced: if hit the runtime limit, limit len(sequence of Local complementations)
    lc_method=None,  # input
    noise_simulation=False,  # input
    noise_model_mapping="depolarizing",  # input
    depolarizing_rate=0.005,  # input
    error_margin=0.1,  # input
    confidence=1,  # input
    mc_map=None,  # advanced*: pass a manual noise map for the monte carlo simulations
    n_cores=8,  # advanced: change if processor has different number of cores
    seed=1,  # input
    graph_type="linear",  # input
    graph_size=2,  # input
    verbose=False,
    save_openqasm="none")

# settings = user_input.setting
# solver = user_input.solver
# nx.draw_networkx(user_input.target_graph, with_labels=True)
# plt.show()
# ray.init()
# start0 = time.time()
# d = solver.solve()
# end0 = time.time()
# ray.shutdown()
# print("runtime0 = ", end0 - start0)
# print("Monte Carlo",
#       [(param, settings.monte_carlo_params[param]) for param in ["n_sample", "n_single", "n_parallel", "seed"]])


# %%
# results
def orbit_analyzer(graph, dir_name='new_orbit', n_lc=None, graph_met_list=None, circ_met_list=None, plots=False, lc_method=None):
    """
    The function take a graph and returns the solver result object containing graph and circuit metrics
    :param graph_met_list: list of graph metrics to consider for correlations. If none given, a default list is used.
    :param circ_met_list: list of circuit metrics to consider for correlations. If none given, a default list is used.
    :param n_lc: number of LC graphs to search for. If none, the whole orbit is targeted.
    :param graph: The input graph. The node labels do not change through the process.
    :param dir_name: the directory to save the resulting graph in
    :param plots: if True the correlation plots are created and saved in the directory given in dir_name
    :return: result object and a list of correlation figures are also saves in the given directory
    """
    user_input = InputParams(
        n_ordering=1,  # input
        rel_inc_thresh=0.2,  # advanced: (0,1) The closer to 0 the closer we get to an exhaustive search for reordering.
        allow_exhaustive=True,  # advanced*: only reason to deactivate is to save runtime if this is the bottleneck
        n_lc_graphs=n_lc,  # input
        lc_method=lc_method,  # input
        noise_simulation=False,  # input
        noise_model_mapping="depolarizing",  # input
        depolarizing_rate=0.005,  # input
        error_margin=0.1,  # input
        confidence=1,  # input
        n_cores=8,  # advanced: change if processor has different number of cores
        seed=1,  # input
        graph_type="nx",  # input
        graph_size=graph,  # input
        verbose=False)
    solver = user_input.solver
    solver.solve()
    res = solver.result
    if graph_met_list is None:
        graph_met_list = ["global_efficiency", "n_edges", "mean_nei_deg", "node_connect", "avg_shortest_path",
                          "max_between", "cluster"]
    if circ_met_list is None:
        circ_met_list = ["n_emitters", "n_cnots", "max_emit_depth", "max_emit_reset_depth", "max_emit_eff_depth",
                         "depth", "std of score", "n_measurements", "n_unitary"]
    res = result_maker(res, graph_met_list=graph_met_list, circ_met_list=circ_met_list)
    if plots:
        plot_figs(res, dir_name=dir_name, graph_mets=graph_met_list, circ_mets=None)
    return res


def result_maker(result, graph_met_list=[], circ_met_list=[]):
    # default properties: "g": alternative graph, "score": infidelity, "map": order permutation map
    # print("number of cases found:", len(result))
    # print("The infidelity", [x[1]["score"] for x in d])

    # print("RunTime: ", end1 - start1, )  # "\nStabiliTime: ", end0-start0)
    # print("1-p method inFidelity:", 1 - (1 - s.depolarizing_rate) ** n_noisy_gate(d[0][0]))

    # graph_met_list = ["node_connect", "cluster", "local_efficiency", "global_efficiency", "max_between", "max_close",
    #                   "min_close", "mean_nei_deg", "edge_connect", "assort", "radius", "diameter", "periphery",
    #                   "center", "pop", "max_deg", "n_edges", "avg_shortest_path"]  # input
    # circ_met_list = ["n_emitters", "n_cnots", "max_emit_depth", "max_emit_eff_depth", "depth", "std of score"]
    if graph_met_list == []:
        graph_met_list = ["global_efficiency", "n_edges", "mean_nei_deg", "node_connect", "avg_shortest_path",
                          "max_between", "cluster"]
    if circ_met_list == []:
        circ_met_list = ["n_emitters", "n_cnots", "max_emit_depth", "max_emit_reset_depth", "max_emit_eff_depth",
                         "depth", "std of score", "n_measurements", "n_unitary"]
    for met in circ_met_list:
        result.add_properties(met)
    # circ metrics calculations
    # for depth calculation purposes we need to remove identity gates first
    unwrapped_circ = [c.copy() for c in result["circuit"]]
    max_emit_depth = []
    max_emit_reset_depth = []
    max_emit_eff_depth = []
    depth = []
    n_emitters = []
    n_cnots = []
    n_measure = []
    n_unitary = []
    for c in unwrapped_circ:
        c.unwrap_nodes()
        c.remove_identity()
        # calculate emitter depths
        e_depth = {}
        reset_depths = {}
        eff_depth = {}
        for e_i in range(c.n_emitters):
            if "max_emit_depth" in circ_met_list:
                e_depth[e_i] = len(c.reg_gate_history(reg=e_i)[1]) - 2
            if "max_emit_reset_depth" in circ_met_list:
                m_list = []  # list of indices of measurement nodes in emitters gate history
                for i, oper in enumerate(c.reg_gate_history(reg=e_i)[0]):
                    # first find a list of nodes in DAG corresponding to measurements
                    if type(oper).__name__ in ['Input', 'MeasurementCNOTandReset', 'Output']:
                        m_list.append(i)
                reset_intervals = [m_list[j + 1] - m_list[j] for j in range(len(m_list) - 1)]
                reset_depths[e_i] = max(reset_intervals)
            # find the max topological depth between two consecutive measurements on the same emitter
            if "max_emit_eff_depth" in circ_met_list:
                node_list = []
                for i, oper in enumerate(c.reg_gate_history(reg=e_i)[0]):
                    # first find a list of nodes in DAG corresponding to measurements
                    if type(oper).__name__ in ['Input', 'MeasurementCNOTandReset', 'Output']:
                        node_list.append(c.reg_gate_history(reg=e_i)[1][i])
                node_depth_list = [c._max_depth(n) for n in node_list]
                depth_diff = [node_depth_list[j + 1] - node_depth_list[j] for j in range(len(node_list) - 1)]
                eff_depth[e_i] = max(depth_diff)
        if "max_emit_depth" in circ_met_list:
            max_emit_depth.append(max(e_depth.values()))
        if "max_emit_reset_depth" in circ_met_list:
            max_emit_reset_depth.append(max(reset_depths.values()))
        if "max_emit_eff_depth" in circ_met_list:
            max_emit_eff_depth.append(max(eff_depth.values()))
        if "depth" in circ_met_list:
            depth.append(max(c.register_depth["e"]))
        # calculate n_emitter and n_cnots
        if "n_emitters" in circ_met_list:
            n_emitters.append(c.n_emitters)
        if "n_unitary" in circ_met_list:
            n_u = 0
            for label in ["SigmaX", "SigmaX", "SigmaX", "Phase", "Hadamard", "CNOT"]:
                if label in c.node_dict:
                    n_u += len(c.get_node_by_labels([label]))
            n_unitary.append(n_u)
        if "n_measurements" in circ_met_list:
            n_measure.append(len(c.get_node_by_labels(["MeasurementCNOTandReset"])))
        if "n_cnots" in circ_met_list:
            if "Emitter-Emitter" in c.node_dict:
                n_cnots.append(len(c.get_node_by_labels(["Emitter-Emitter", "CNOT"])))
            else:
                n_cnots.append(0)
    if "n_emitters" in circ_met_list:
        result["n_emitters"] = n_emitters
    if "n_cnots" in circ_met_list:
        result["n_cnots"] = n_cnots
    if "n_unitary" in circ_met_list:
        result["n_unitary"] = n_unitary
    if "n_measurements" in circ_met_list:
        result["n_measurements"] = n_measure
    if "max_emit_depth" in circ_met_list:
        result["max_emit_depth"] = max_emit_depth
    if "max_emit_reset_depth" in circ_met_list:
        result["max_emit_reset_depth"] = max_emit_reset_depth
    if "max_emit_eff_depth" in circ_met_list:
        result["max_emit_eff_depth"] = max_emit_eff_depth
    if "depth" in circ_met_list:
        result["depth"] = depth
    # result["std of score"] = [np.std(x.all_scores) for x in solver.mc_list] if settings.monte_carlo else len(result) * [
    #     0]

    # graph metric:
    result.add_properties("graph_metric")
    # a bit complicated: for each index in the result object, there is a corresponding dict of graph_metrics.
    # for each graph, this dict contains the metric values for the metrics selected by user: {"met1": val1, ...} for each g
    result["graph_metric"] = [dict() for _ in range(len(result))]

    for met in graph_met_list:
        for i, g in enumerate(result["g"]):
            result["graph_metric"][i][met] = graph_met_value(met, g)
    return result


# result = solver.result
# result = result_maker(result)


# %% Plot
def plot_figs(result, indices=None, dir_name='new', graph_mets=None, circ_mets=None):
    circ_mets = ["n_cnots", "max_emit_eff_depth", "score"] if circ_mets is None else circ_mets
    if graph_mets is None:
        graph_mets = ["global_efficiency", "n_edges", "mean_nei_deg", "node_connect", "avg_shortest_path",
                      "max_between", "cluster"]
    for c_met in circ_mets:
        met_hist(result, c_met, show_plot=False, index_list=indices, dir_name=dir_name)
        for g_met in graph_mets:
            met_met(result, c_met, g_met, show_plot=False, index_list=indices, dir_name=dir_name)


# %% correlations
def correlation_checker(result, list_mets1, list_mets2):
    """
    :param result: result object
    :param list_mets1: list of circuit metrics
    :param list_mets2: list of grpah metrics
    :return: all combination of graph and circuit metrics linear correlations will be calculated
    """
    for met1 in list_mets1:
        for met2 in list_mets2:
            x = [g_met[f'{met1}'] for g_met in result['graph_metric']] if met1 in result['graph_metric'][0] else result[
                f'{met1}']
            y = [g_met[f'{met2}'] for g_met in result['graph_metric']] if met2 in result['graph_metric'][0] else result[
                f'{met2}']
            corr_s, p_value_s = spearmanr(x, y)
            print(f'Spearmans {met1}-{met2}: %.3f' % corr_s, round(p_value_s, 3))
            corr_p, p_value_p = pearsonr(x, y)
            print(f'Pearsons {met1}-{met2}: %.3f' % corr_p, round(p_value_p, 3))


def corr_with_mean(list1, list2, print_result = True):
    """
    given two lists of data corresponding to each other element by element, it finds the correlation between list1 with
    mean values of list 2. For instance, the average number of cnots corresponding to each case of edge count in graph.
    :param list1: to be used as reference
    :param list2: to be averaged over for each case
    :return: pearson correlation coefficient
    """
    corr_dict = dict()
    for i, x in enumerate(list1):
        if corr_dict.get(x):
            corr_dict[x].append(list2[i])
        else:
            corr_dict[x] = [list2[i]]
    corr_list1 = list(corr_dict.keys())
    corr_list2 = [np.mean(val) for val in corr_dict.values()]
    corr_list2_std = [np.std(val) for val in corr_dict.values()]
    corr_p, p_value_p = pearsonr(corr_list1, corr_list2)
    if print_result:
        print(f'Pearsons List1 vs Mean_List2: %.3f' % corr_p, round(p_value_p, 3))
    return corr_p, (corr_list1, corr_list2, corr_list2_std)


# %% Pre-Post processing
# metrics to measure:
"fidelity, CNOT count, max emitter depth (corrected), emitter effective depth, circuit depth (corrected)"
# separate the circuits with same relabeling map to be analyzed alone
result = {"map": [{0: 0}]}
maps_list = list(set([tuple(result["map"][i].items()) for i in range(len(result))]))
# a dictionary between each relabel map and a list of indices in result corresponding to that map
index_map_dict = dict(zip(maps_list, [[] for _ in maps_list]))
for i in range(len(result)):
    index_map_dict[tuple(result["map"][i].items())].append(i)


def plot_map_based():
    for i, index_list in enumerate(index_map_dict.values()):
        plot_figs(indices=index_list, dir_name=f'map_{i}_len{len(index_list)}')


def plot_graphs(result):
    for i, g in enumerate(result['g']):
        nx.draw_networkx(g, with_labels=1, pos=nx.kamada_kawai_layout(g))
        plt.figtext(0.10, 0.05, f"# of CNOTs {result['n_cnots'][i]}")
        plt.figtext(0.30, 0.05, f"# of emitters {result['n_emitters'][i]}")
        plt.figtext(0.50, 0.05, f"fidelity {round(1 - result['score'][i], 3)}")
        plt.figtext(0.70, 0.05, f"max E-depth {result['max_emit_eff_depth'][i]}")
        plt.show()


# plot_figs()
# plot_graphs()
# plot_map_based()

# %% analyze
# corr = GraphCorr(graph_list=g_list)
# corr._graph_circ_dictionary = dict(zip(g_list, circ_list))
# corr.met_distribution(met="max_emit_depth", show_plot=True)
# # %%
#
# corr.met_distribution(met="max_emit_depth", show_plot=True)
# corr.met_distribution(met="cnot", show_plot=True)
# corr.met_met("cnot", "depth", n=None, p=None, show_plot=True)
#
# sort_by_emit = sorted(circ_list, key=lambda x: x[0].n_emitters)
# min_min_emit_cnot = sorted(circ_list, key=lambda x: len(x[0].get_node_by_labels(["Emitter-Emitter", "CNOT"])))
# min_min_emit_depth = sorted(circ_list, key=lambda x: x[0].depth)
# min_min_emit_cnot[0][0].draw_circuit()
# min_min_emit_depth[0][0].draw_circuit()
# circ_list[0][0].draw_circuit()
# nx.draw_networkx(min_min_emit_cnot[0][1]['g'], with_labels=1)
# nx.draw_networkx(min_min_emit_depth[0][1]['g'], with_labels=1)
# plt.show()


# %%
# ll=[(i,result['max_emit_eff_depth'][i]) for i, x in enumerate(result['n_cnots']) if x==2]
# ll.sort(key=lambda x: x[1])
# print([result['n_emitters'][i] for i, x in enumerate(result['n_cnots']) if x==2])
# %%
def _edge2adj(e_list):
    assert len(e_list) % 2 == 0
    edges = [(e_list[2 * i], e_list[2 * i + 1]) for i in range(int(len(e_list) / 2))]
    edges.sort()
    edges = [tuple(sorted(list(x))) for x in edges]
    edges.sort()
    g1 = nx.Graph()
    g1.add_nodes_from(range(max(e_list)))
    g1.add_edges_from(edges)
    return nx.to_numpy_array(g1)


# %%
edge_seq = []


def lcs(edge_seq, method="lc_with_iso", n_lc_graphs=None, seed=1):
    adj1 = _edge2adj(edge_seq)
    user_input0 = InputParams(
        n_ordering=1,  # input
        rel_inc_thresh=0.2,  # advanced: (0,1) The closer to 0 the closer we get to an exhaustive search for reordering.
        allow_exhaustive=True,  # advanced*: only reason to deactivate is to save runtime if this is the bottleneck
        iso_thresh=None,  # advanced: if not enough relabeled graphs are found, set it to a larger number!
        n_lc_graphs=n_lc_graphs,  # input
        lc_orbit_depth=None,  # advanced: if hit the runtime limit, limit len(sequence of Local complementations)
        lc_method=method,  # input
        noise_simulation=False,  # input
        noise_model_mapping="depolarizing",  # input
        depolarizing_rate=0.005,  # input
        error_margin=0.1,  # input
        confidence=0.99,  # input
        mc_map=None,  # advanced*: pass a manual noise map for the monte carlo simulations
        n_cores=8,  # advanced: change if processor has different number of cores
        seed=seed,  # input
        graph_type="adj",  # input
        graph_size=adj1,  # input
        verbose=False,
        save_openqasm="none")

    solver = user_input0.solver
    nx.draw_networkx(user_input0.target_graph, with_labels=True)
    # plt.show()
    # plt.close()
    solver.solve()
    result0 = solver.result
    result0 = result_maker(result0)
    lc_list = [g for g in result0['g']]
    lc_adjs = [nx.to_numpy_array(g) for g in lc_list]
    return lc_adjs


def find_best(adj1, file_name="case1", dir_name="new", conf=0.99, n_reordering=None, circ_met_list=[], seed=1):
    nn = np.shape(adj1)[0]
    max_relabel = factorial(nn) if n_reordering is None else n_reordering
    user_input = InputParams(
        n_ordering=max_relabel,  # input
        rel_inc_thresh=0.2,  # advanced: (0,1) The closer to 0 the closer we get to an exhaustive search for reordering.
        allow_exhaustive=True,  # advanced*: only reason to deactivate is to save runtime if this is the bottleneck
        iso_thresh=None,  # advanced: if not enough relabeled graphs are found, set it to a larger number!
        n_lc_graphs=1,  # input
        lc_orbit_depth=1,  # advanced: if hit the runtime limit, limit len(sequence of Local complementations)
        lc_method=None,  # input
        noise_simulation=conf,  # input
        noise_model_mapping="depolarizing",  # input
        depolarizing_rate=0.005,  # input
        error_margin=0.05,  # input
        confidence=conf,  # input
        mc_map=None,  # advanced*: pass a manual noise map for the monte carlo simulations
        n_cores=8,  # advanced: change if processor has different number of cores
        seed=seed,  # input
        graph_type="adj",  # input
        graph_size=adj1,  # input
        verbose=False,
        save_openqasm="none")
    solver = user_input.solver
    settings = user_input.setting
    solver.solve()
    result = solver.result
    result = result_maker(result, graph_met_list=['n_edges'], circ_met_list=circ_met_list)
    out_dict = (result._data).copy()
    # out_dict['fidelity'] = [round(1 - s, 5) for s in out_dict['score']]
    out_dict['edges'] = [list(g.edges) for g in out_dict['g']]
    del out_dict['g']
    del out_dict['score']
    del out_dict['circuit']
    # del out_dict['std of score']
    del out_dict['graph_metric']
    df = pd.DataFrame(out_dict)
    new_path = f'/Users/sobhan/Desktop/EntgClass/{dir_name}'
    if not os.path.exists(new_path):
        os.makedirs(new_path)
    filename = new_path + f'/{file_name}.csv'
    df.to_csv(filename, index=False)
    result.save2json(new_path, f'res_{file_name}')
    cost = []
    for i in range(len(result)):
        cost.append(result['n_emitters'][i] * 100 + result['n_cnots'][i] * 10 + result['max_emit_reset_depth'][i])
    out_dict['cost'] = cost
    return out_dict


def graph_analyzer(edge_seq, graph_class: str, method="lc_with_iso", conf=1, n_lc_graphs=None, n_reordering=None,
                   circ_met_list=[], seed=1):
    """
    exhaustive graph analyzer given the flattened edge list
    no correlation analysis involved but cost analysis based on circuit metrics is included
    :param edge_seq:
    :param graph_class:
    :param method:
    :param conf: confidence for noise simulation
    :return:
    """
    lc_adjs = lcs(edge_seq, method=method, n_lc_graphs=n_lc_graphs, seed=seed)  ###
    print("number of LC graphs = ", len(lc_adjs))
    # relabeled_info = {'index': list(range(len(n_es))), 'map': maps, 'n_emitters': n_es}###
    # df = pd.DataFrame(relabeled_info)
    new_path = f'/Users/sobhan/Desktop/EntgClass/{graph_class}'
    if not os.path.exists(new_path):
        os.makedirs(new_path)
    # file_name = new_path + f'/iso_info.csv'
    # df.to_csv(file_name, index=False)
    best_cost = ("", 10E6, [])  # 1st element is which iso and lc is this graph, 2nd is the cost, 3rd is the edge list
    best_fid = ("", 0, [0])
    worst_cost = ("", 0, [])
    worst_fid = ("", 1, [0])
    best_ncnot = ("", 10E6, [])
    worst_ncnot = ("", 0, [])
    best_depth = ("", 10E6, [])
    worst_depth = ("", 0, [])
    num_iso_list = []
    for i, adj in enumerate(lc_adjs):
        out_dict = find_best(adj, file_name=f"case{i}", dir_name=graph_class, conf=conf, n_reordering=n_reordering,
                             circ_met_list=circ_met_list, seed=seed)
        num_iso_list.append(len(out_dict['cost']))
        for j, edge_list in enumerate(out_dict['edges']):
            if out_dict['cost'][j] < best_cost[1]:
                best_cost = (f"LC: {i} iso: {j}", out_dict['cost'][j], edge_list)
            if out_dict['cost'][j] > worst_cost[1]:
                worst_cost = (f"LC: {i} iso: {j}", out_dict['cost'][j], edge_list)
            if out_dict['n_cnots'][j] < best_ncnot[1]:
                best_ncnot = (f"LC: {i} iso: {j}", out_dict['n_cnots'][j], edge_list)
            if out_dict['n_cnots'][j] > worst_ncnot[1]:
                worst_ncnot = (f"LC: {i} iso: {j}", out_dict['n_cnots'][j], edge_list)
            if out_dict['max_emit_reset_depth'][j] < best_depth[1]:
                best_depth = (f"LC: {i} iso: {j}", out_dict['max_emit_reset_depth'][j], edge_list)
            if out_dict['max_emit_reset_depth'][j] > worst_depth[1]:
                worst_depth = (f"LC: {i} iso: {j}", out_dict['max_emit_reset_depth'][j], edge_list)
            # if out_dict['fidelity'][j] > best_fid[1]:
            #     best_fid = (f"LC: {i} iso: {j}", out_dict['fidelity'][j], edge_list)
            # if out_dict['fidelity'][j] < best_fid[1]:
            #     worst_fid = (f"LC: {i} iso: {j}", out_dict['fidelity'][j], edge_list)
    print("number of LC graphs = ", len(lc_adjs))
    print("number of iso found = ", num_iso_list, f"\ntotal number of cases = {sum(num_iso_list)}")

    text = f"number of LC graphs = {len(lc_adjs)}\nnumber of iso found = {num_iso_list}" \
           f"\nbest n cnot {best_ncnot[1]}: {best_ncnot[0]}\nworst n cnot {worst_ncnot[1]}: {worst_ncnot[0]}" \
           f"\nbest depth {best_depth[1]}: {best_depth[0]}\nworst depth {worst_depth[1]}: {worst_depth[0]}" \
           f"\ntotal number of cases = {sum(num_iso_list)}" \
           f"\nLC method {method}, none exhaustive? LC {n_lc_graphs}, n_order {n_reordering}"
        # f"\nbest graph w.r.t cost {best_cost[0]}\nbest graph w.r.t fidelity {best_fid[0]}" \
        # f"\nworst graph w.r.t cost {worst_cost[0]}\nworst graph w.r.t fidelity {worst_fid[0]}" \
    filename = new_path + f'/bests.txt'
    with open(filename, "w") as file:
        file.write(text)
    g = nx.from_numpy_array(_edge2adj(edge_seq))
    fig = plt.figure(figsize=(4, 3), dpi=150)
    nx.draw_networkx(g, pos=nx.kamada_kawai_layout(g))
    plt.savefig(new_path + f'/graph.png')
    plt.close(fig)
    g1 = nx.Graph()
    g1.add_edges_from(best_cost[2])
    fig = plt.figure(figsize=(4, 3), dpi=150)
    nx.draw_networkx(g1, pos=nx.kamada_kawai_layout(g1))
    plt.savefig(new_path + f'/best_cost.png')
    plt.close(fig)
    # g2 = nx.Graph()
    # g2.add_edges_from(best_fid[2])
    # fig = plt.figure(figsize=(4, 3), dpi=150)
    # nx.draw_networkx(g2, pos=nx.kamada_kawai_layout(g2))
    # plt.savefig(new_path + f'/best_fid.png')
    # plt.close(fig)
    return best_fid, best_cost, best_ncnot, best_depth, worst_fid, worst_cost, worst_ncnot, worst_depth


# %%
def LC_scaling_test(g, graph_class: str, method="random_with_iso", conf=0, n_lc_list=[*(range(1, 10))], n_reordering=1):
    edge_list = [*g.edges]
    edges = [x for sublist in edge_list for x in sublist]
    bw = []  # best worst cases
    avgs_cnots = []
    avgs_depths = []
    stds_cnots = []
    stds_depths = []
    for i, n in enumerate(n_lc_list):
        bw.append([])
        for j in range(50):
            bw[i].append(graph_analyzer(edges, f"{graph_class}/LC{n}/T{j}", method=method, conf=conf,
                                        n_lc_graphs=n, n_reordering=n_reordering))
        bw_cnot_i = [(x[2][1], x[6][1]) for x in bw[i]]  # best and worst number of CNOTS
        bw_depth_i = [(x[3][1], x[7][1]) for x in bw[i]]  # best and worst depth
        avg_c = tuple(np.mean(x) for x in zip(*bw_cnot_i))
        avg_d = tuple(np.mean(x) for x in zip(*bw_depth_i))

        std_c = tuple(np.std(x) for x in zip(*bw_cnot_i))
        std_d = tuple(np.std(x) for x in zip(*bw_depth_i))

        avgs_cnots.append(avg_c)
        avgs_depths.append(avg_d)
        stds_cnots.append(std_c)
        stds_depths.append(std_d)

    df_cnot = pd.DataFrame((avgs_cnots, stds_cnots))
    df_depth = pd.DataFrame((avgs_depths, stds_depths))

    output_path1 = f'/Users/sobhan/Desktop/EntgClass/{graph_class}/bwCNOT.csv'
    df_cnot.to_csv(output_path1, index=False, header=False)
    output_path2 = f'/Users/sobhan/Desktop/EntgClass/{graph_class}/bwDepth.csv'
    df_depth.to_csv(output_path2, index=False, header=False)
    return avgs_cnots, stds_cnots, avgs_depths, stds_depths, bw


def iso_scaling_test(g, graph_class: str, method=None, conf=0, n_lc_graphs=1, n_iso_list=[*(range(1, 10))]):
    adj_matrix = nx.to_numpy_array(g)
    nodes = [*g.nodes]
    bw = []  # best worst cases
    avgs_cnots = []
    avgs_depths = []
    stds_cnots = []
    stds_depths = []
    for i, n in enumerate(n_iso_list):
        bw.append([])
        for j in range(50):
            # randomize each initial graph
            np.random.shuffle(nodes)
            new_adj = relabel(adj_matrix, nodes)
            g = nx.from_numpy_array(new_adj)
            edge_list = [*g.edges]
            edges = [x for sublist in edge_list for x in sublist]

            bw[i].append(graph_analyzer(edges, f"{graph_class}/ISO{n}/T{j}", method=method, conf=conf,
                                        n_lc_graphs=n_lc_graphs, n_reordering=n))
        bw_cnot_i = [(x[2][1], x[6][1]) for x in bw[i]]  # best and worst number of CNOTS
        bw_depth_i = [(x[3][1], x[7][1]) for x in bw[i]]  # best and worst depth
        avg_c = tuple(np.mean(x) for x in zip(*bw_cnot_i))
        avg_d = tuple(np.mean(x) for x in zip(*bw_depth_i))

        std_c = tuple(np.std(x) for x in zip(*bw_cnot_i))
        std_d = tuple(np.std(x) for x in zip(*bw_depth_i))

        avgs_cnots.append(avg_c)
        avgs_depths.append(avg_d)
        stds_cnots.append(std_c)
        stds_depths.append(std_d)

    df_cnot = pd.DataFrame((avgs_cnots, stds_cnots))
    df_depth = pd.DataFrame((avgs_depths, stds_depths))

    output_path1 = f'/Users/sobhan/Desktop/EntgClass/{graph_class}/bwCNOT.csv'
    df_cnot.to_csv(output_path1, index=False, header=False)
    output_path2 = f'/Users/sobhan/Desktop/EntgClass/{graph_class}/bwDepth.csv'
    df_depth.to_csv(output_path2, index=False, header=False)
    return avgs_cnots, stds_cnots, avgs_depths, stds_depths, bw


def LC_scaling_known(result, graph_class: str, n_lc_list=None):
    if n_lc_list is None:
        n_lc_list = [*(range(1, 10))]
    b = len(result)
    c_means = []
    c_stds = []
    d_means = []
    d_stds = []
    for n_lc in n_lc_list:
        c_range = []
        d_range = []
        for j in range(50):
            random_integers = np.random.randint(0, b, n_lc)
            cnots = [result['n_cnots'][i] for i in random_integers]
            depth = [result['max_emit_eff_depth'][i] for i in random_integers]
            c_range.append(max(cnots) - min(cnots))
            d_range.append(max(depth) - min(depth))
        c_means.append((0, np.mean(c_range)))
        c_stds.append((np.std(c_range), np.std(c_range)))
        d_means.append((0, np.mean(d_range)))
        d_stds.append((np.std(d_range), np.std(d_range)))
    df_cnot = pd.DataFrame((c_means, c_stds))
    df_depth = pd.DataFrame((d_means, d_stds))

    output_path1 = f'/Users/sobhan/Desktop/EntgClass/{graph_class}/bwCNOT.csv'
    df_cnot.to_csv(output_path1, index=False, header=False)
    output_path2 = f'/Users/sobhan/Desktop/EntgClass/{graph_class}/bwDepth.csv'
    df_depth.to_csv(output_path2, index=False, header=False)
    return c_means, c_stds, d_means, d_stds


def rgs_analysis(res_rgsn, filename: str):
    n_cnots = []
    for c in res_rgsn['circuit']:
        if "Emitter-Emitter" in c.node_dict:
            n_cnots.append(len(c.get_node_by_labels(["Emitter-Emitter", "CNOT"])))
        else:
            n_cnots.append(0)
    res_rgsn.add_properties("n_cnot")
    res_rgsn["n_cnots"] = n_cnots
    graph_met_list = ["node_connect", "cluster", "local_efficiency", "global_efficiency", "max_between", "max_close",
                      "min_close", "mean_nei_deg", "edge_connect", "radius", "diameter", "periphery",
                      "center", "pop", "max_deg", "n_edges", "avg_shortest_path", "assort"]
    res_rgsn.add_properties("graph_metric")
    res_rgsn["graph_metric"] = [dict() for _ in range(len(res_rgsn))]
    for met in graph_met_list:
        for i, g in enumerate(res_rgsn["g"]):
            res_rgsn["graph_metric"][i][met] = graph_met_value(met, g)
    res_rgsn.save2json(f"/Users/sobhan/Desktop/EntgClass/RGS/{filename}", f'{filename}')
    correlation_checker(res_rgsn, ['n_cnots'], graph_met_list)
    plot_figs(res_rgsn, indices=None, dir_name=filename, graph_mets=graph_met_list, circ_mets=['n_cnots'])


def rnd_graph_orbit_cnots(size, number_of_graphs):
    list_cnot_list = []
    restuls_list = []
    rng = np.random.default_rng()
    rnd_seeds = rng.integers(low=1, high=1000 * number_of_graphs, size=number_of_graphs)
    for i in rnd_seeds:
        user_input0 = InputParams(
            n_ordering=1,  # input
            rel_inc_thresh=0.2,
            # advanced: (0,1) The closer to 0 the closer we get to an exhaustive search for reordering.
            allow_exhaustive=True,  # advanced*: only reason to deactivate is to save runtime if this is the bottleneck
            iso_thresh=None,  # advanced: if not enough relabeled graphs are found, set it to a larger number!
            n_lc_graphs=None,  # input
            lc_orbit_depth=None,  # advanced: if hit the runtime limit, limit len(sequence of Local complementations)
            lc_method="lc_with_iso",  # input
            noise_simulation=False,  # input
            noise_model_mapping="depolarizing",  # input
            depolarizing_rate=0.005,  # input
            error_margin=0.1,  # input
            confidence=0.99,  # input
            mc_map=None,  # advanced*: pass a manual noise map for the monte carlo simulations
            n_cores=8,  # advanced: change if processor has different number of cores
            seed=i,  # input
            graph_type="rnd",  # input
            graph_size=size,  # input
            verbose=False,
            save_openqasm="none")
        solver = user_input0.solver
        solver.solve()
        result0 = solver.result
        result0.add_properties("n_cnots")
        n_cnots = []
        for c in result0['circuit']:
            if "Emitter-Emitter" in c.node_dict:
                n_cnots.append(len(c.get_node_by_labels(["Emitter-Emitter", "CNOT"])))
            else:
                n_cnots.append(0)
        result0["n_cnots"] = n_cnots
        restuls_list.append(result0)
        list_cnot_list.append(n_cnots)
    return list_cnot_list, restuls_list
# %%
# graph_analyzer([0,1,1,2,2,3,3,4,4,5,5,0,0,2,5,3,1,4], "class 19", method=None, conf=1)
# LC_scaling_test(g,"random_8_05", method="random_with_iso", conf=0, n_lc_list=[*(range(2, 10))], n_reordering=1)
