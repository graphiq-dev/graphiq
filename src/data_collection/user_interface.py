###
# data collection script #
###
import time
import pandas as pd
import os
import json

import networkx as nx
import numpy as np
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
from src.utils.relabel_module import iso_finder, emitter_sorted, lc_orbit_finder, get_relabel_map
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

from benchmarks.graph_states import linear_cluster_state, repeater_graph_states, lattice_cluster_state

import ray

from src.data_collection.ui_functions import *
from correlation_module import *

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
    n_ordering=5000,  # input
    rel_inc_thresh=0.2,  # advanced: (0,1) The closer to 0 the closer we get to an exhaustive search for reordering.
    allow_exhaustive=True,  # advanced*: only reason to deactivate is to save runtime if this is the bottleneck
    iso_thresh=None,  # advanced: if not enough relabeled graphs are found, set it to a larger number!
    n_lc_graphs=1,  # input
    lc_orbit_depth=None,  # advanced: if hit the runtime limit, limit len(sequence of Local complementations)
    lc_method="",  # input
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

settings = user_input.setting
solver = user_input.solver
nx.draw_networkx(user_input.target_graph, with_labels=True)
plt.show()

ray.init()
start0 = time.time()
# d = solver.solve()
end0 = time.time()
ray.shutdown()
print("runtime0 = ", end0 - start0)
print("Monte Carlo",
      [(param, settings.monte_carlo_params[param]) for param in ["n_sample", "n_single", "n_parallel", "seed"]])


# %%
# results


def result_maker(result):
    # default properties: "g": alternative graph, "score": infidelity, "map": order permutation map
    # print("number of cases found:", len(result))
    # print("The infidelity", [x[1]["score"] for x in d])

    # print("RunTime: ", end1 - start1, )  # "\nStabiliTime: ", end0-start0)
    # print("1-p method inFidelity:", 1 - (1 - s.depolarizing_rate) ** n_noisy_gate(d[0][0]))

    result.add_properties("n_emitters")
    result.add_properties("n_cnots")
    result.add_properties("max_emit_depth")
    result.add_properties("depth")
    result.add_properties("std of score")
    # circ metrics calculations
    # for depth calculation purposes we need to remove identity gates first
    unwrapped_circ = [c.copy() for c in result["circuit"]]
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

    result["n_emitters"] = n_emitters
    result["n_cnots"] = n_cnots
    result["max_emit_depth"] = max_emit_depth
    result["max_emit_eff_depth"] = max_emit_eff_depth
    result["depth"] = depth
    # result["std of score"] = [np.std(x.all_scores) for x in solver.mc_list] if settings.monte_carlo else len(result) * [
    #     0]

    # graph metric:
    result.add_properties("graph_metric")
    # a bit complicated: for each index in the result object, there is a corresponding dict of graph_metrics.
    # for each graph, this dict contains the metric values for the metrics selected by user: {"met1": val1, ...} for each g
    result["graph_metric"] = [dict() for _ in range(len(result))]
    graph_met_list = ["node_connect", "cluster", "local_efficiency", "global_efficiency", "max_between", "max_close",
                      "min_close",
                      "mean_nei_deg", "edge_connect", "assort", "radius", "diameter", "periphery", "center"]  # input
    graph_met_list = []
    for met in graph_met_list:
        for i, g in enumerate(result["g"]):
            result["graph_metric"][i][met] = graph_met_value(met, g)
    return result


# result = solver.result
# result = result_maker(result)


# %% Plot

def plot_figs(result, indices=None, dir_name='new'):
    met_hist(result, "score", show_plot=False, index_list=indices, dir_name=dir_name)
    met_hist(result, "n_cnots", show_plot=False, index_list=indices, dir_name=dir_name)
    met_hist(result, "depth", show_plot=False, index_list=indices, dir_name=dir_name)
    met_met(result, "n_cnots", "score", show_plot=False, index_list=indices, dir_name=dir_name)

    met_met(result, "n_cnots", "mean_nei_deg", show_plot=False, index_list=indices, dir_name=dir_name)
    met_met(result, "n_cnots", "node_connect", show_plot=False, index_list=indices, dir_name=dir_name)
    met_met(result, "n_cnots", "edge_connect", show_plot=False, index_list=indices, dir_name=dir_name)
    met_met(result, "n_cnots", "assort", show_plot=False, index_list=indices, dir_name=dir_name)
    met_met(result, "n_cnots", "radius", show_plot=False, index_list=indices, dir_name=dir_name)
    met_met(result, "n_cnots", "diameter", show_plot=False, index_list=indices, dir_name=dir_name)
    met_met(result, "n_cnots", "periphery", show_plot=False, index_list=indices, dir_name=dir_name)
    met_met(result, "n_cnots", "center", show_plot=False, index_list=indices, dir_name=dir_name)
    met_met(result, "n_cnots", "cluster", show_plot=False, index_list=indices, dir_name=dir_name)


# %% Pre-Post processing
# metrics to measure:
"fidelity, CNOT count, max emitter depth (corrected), emitter effective depth, circuit depth (corrected)"
# separate the circuits with same relabeling map to be analyzed alone
result = {"map": [0]}
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
    g1 = nx.Graph()
    g1.add_edges_from(edges)
    return nx.to_numpy_array(g1)


# %%
edge_seq = []


def lcs(edge_seq, method="lc_with_iso"):
    adj1 = _edge2adj(edge_seq)
    user_input0 = InputParams(
        n_ordering=1,  # input
        rel_inc_thresh=0.2,  # advanced: (0,1) The closer to 0 the closer we get to an exhaustive search for reordering.
        allow_exhaustive=True,  # advanced*: only reason to deactivate is to save runtime if this is the bottleneck
        iso_thresh=None,  # advanced: if not enough relabeled graphs are found, set it to a larger number!
        n_lc_graphs=None,  # input
        lc_orbit_depth=None,  # advanced: if hit the runtime limit, limit len(sequence of Local complementations)
        lc_method=method,  # input
        noise_simulation=False,  # input
        noise_model_mapping="depolarizing",  # input
        depolarizing_rate=0.005,  # input
        error_margin=0.1,  # input
        confidence=1,  # input
        mc_map=None,  # advanced*: pass a manual noise map for the monte carlo simulations
        n_cores=8,  # advanced: change if processor has different number of cores
        seed=1,  # input
        graph_type="adj",  # input
        graph_size=adj1,  # input
        verbose=False,
        save_openqasm="none")

    solver = user_input0.solver
    nx.draw_networkx(user_input0.target_graph, with_labels=True)
    plt.show()
    plt.close()
    solver.solve()
    result0 = solver.result
    result0 = result_maker(result0)
    lc_list = [g for g in result0['g']]
    lc_adjs = [nx.to_numpy_array(g) for g in lc_list]
    return lc_adjs


def find_best(adj1, file_name="case1", dir_name="new", conf=1):
    nn = np.shape(adj1)[0]
    max_relabel = np.math.factorial(nn)
    user_input = InputParams(
        n_ordering=max_relabel,  # input
        rel_inc_thresh=0.2,  # advanced: (0,1) The closer to 0 the closer we get to an exhaustive search for reordering.
        allow_exhaustive=True,  # advanced*: only reason to deactivate is to save runtime if this is the bottleneck
        iso_thresh=None,  # advanced: if not enough relabeled graphs are found, set it to a larger number!
        n_lc_graphs=1,  # input
        lc_orbit_depth=1,  # advanced: if hit the runtime limit, limit len(sequence of Local complementations)
        lc_method="",  # input
        noise_simulation=True,  # input
        noise_model_mapping="depolarizing",  # input
        depolarizing_rate=0.005,  # input
        error_margin=0.05,  # input
        confidence=conf,  # input
        mc_map=None,  # advanced*: pass a manual noise map for the monte carlo simulations
        n_cores=8,  # advanced: change if processor has different number of cores
        seed=1,  # input
        graph_type="adj",  # input
        graph_size=adj1,  # input
        verbose=False,
        save_openqasm="none")
    solver = user_input.solver
    settings = user_input.setting
    solver.solve()
    result = solver.result
    result = result_maker(result)
    out_dict = (result._data).copy()
    out_dict['fidelity'] = [round(1 - s, 5) for s in out_dict['score']]
    out_dict['edges'] = [list(g.edges) for g in out_dict['g']]
    del out_dict['g']
    del out_dict['score']
    del out_dict['circuit']
    del out_dict['std of score']
    del out_dict['graph_metric']
    df = pd.DataFrame(out_dict)
    new_path = f'/Users/sobhan/Desktop/EntgClass/{dir_name}'
    if not os.path.exists(new_path):
        os.makedirs(new_path)
    filename = new_path + f'/{file_name}.csv'
    df.to_csv(filename, index=False)
    cost = []
    for i in range(len(result)):
        cost.append(result['n_emitters'][i] * 100 + result['n_cnots'][i] * 10 + result['max_emit_eff_depth'][i])
    out_dict['cost'] = cost
    return out_dict


def graph_analyzer(edge_seq, graph_class: str, method="lc_with_iso", conf=1):
    lc_adjs = lcs(edge_seq, method=method)###
    print("number of LC graphs = ", len(lc_adjs))
    # relabeled_info = {'index': list(range(len(n_es))), 'map': maps, 'n_emitters': n_es}###
    # df = pd.DataFrame(relabeled_info)
    new_path = f'/Users/sobhan/Desktop/EntgClass/{graph_class}'
    if not os.path.exists(new_path):
        os.makedirs(new_path)
    # file_name = new_path + f'/iso_info.csv'
    # df.to_csv(file_name, index=False)
    best_cost = ("", 10E6, [])  # 1st element is which iso and lc is this graph, 2nd is the cost, 3rd is the edge list
    best_fid = ("", 0, [])
    num_iso_list = []
    for i, adj in enumerate(lc_adjs):
        out_dict = find_best(adj, file_name=f"case{i}", dir_name=graph_class, conf=conf)
        num_iso_list.append(len(out_dict['cost']))
        for j, edge_list in enumerate(out_dict['edges']):
            if out_dict['cost'][j] < best_cost[1]:
                best_cost = (f"LC: {i} iso: {j}", out_dict['cost'][j], edge_list)
            if out_dict['fidelity'][j] > best_fid[1]:
                best_fid = (f"LC: {i} iso: {j}", out_dict['fidelity'][j], edge_list)
    print("number of LC graphs = ", len(lc_adjs))
    print("number of iso found = ", num_iso_list, f"\ntotal number of cases = {sum(num_iso_list)}")

    text = f"number of LC graphs = {len(lc_adjs)}\nnumber of iso found = {num_iso_list}" \
           f"\nbest graph w.r.t cost {best_cost[0]}\nbest graph w.r.t fidelity {best_fid[0]}" \
           f"\ntotal number of cases = {sum(num_iso_list)}"
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
    g2 = nx.Graph()
    g2.add_edges_from(best_fid[2])
    fig = plt.figure(figsize=(4, 3), dpi=150)
    nx.draw_networkx(g2, pos=nx.kamada_kawai_layout(g2))
    plt.savefig(new_path + f'/best_fid.png')
    plt.close(fig)
    return best_fid, best_cost

# %%
graph_analyzer([0,1,1,2,2,3,3,4,4,5,5,0,0,2,5,3,1,4], "class 19", method="", conf=1)