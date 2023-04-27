###
# data collection script #
###
import time

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
user_input = InputParams(
    n_ordering=4,  # input
    rel_inc_thresh=0.2,  # advanced: (0,1) The closer to 0 the closer we get to an exhaustive search for reordering.
    allow_exhaustive=True,  # advanced*: only reason to deactivate is to save runtime if this is the bottleneck
    iso_thresh=None,  # advanced: if not enough relabeled graphs are found, set it to a larger number!
    n_lc_graphs=2,  # input
    lc_orbit_depth=None,  # advanced: if hit the runtime limit, limit len(sequence of Local complementations)
    noise_model_mapping="depolarizing",  # input
    depolarizing_rate=0.03,  # input
    error_margin=0.02,  # input
    confidence=0.90,  # input
    mc_map=None,  # advanced*: pass a manual noise map for the monte carlo simulations
    n_cores=8,  # advanced: change if processor has different number of cores
    seed=None,  # input
    graph_type="draw",  # input
    graph_size=5,  # input
    verbose=False,
    save_openqasm="none")

settings = user_input.setting
solver = user_input.solver
nx.draw_networkx(user_input.target_graph, with_labels=True)
plt.show()

ray.init()
start0 = time.time()
d = solver.solve()
end0 = time.time()
ray.shutdown()

#
start1 = time.time()
end1 = time.time()
# %%
# results
result = solver.result
# default properties: "g": alternative graph, "score": infidelity, "map": order permutation map
#print("number of cases found:", len(result))
# print("The infidelity", [x[1]["score"] for x in d])

#print("RunTime: ", end1 - start1, )  # "\nStabiliTime: ", end0-start0)
# print("1-p method inFidelity:", 1 - (1 - s.depolarizing_rate) ** n_noisy_gate(d[0][0]))
print("Monte Carlo",
      [(param, settings.monte_carlo_params[param]) for param in ["n_sample", "n_single", "n_parallel", "seed"]])

result.sort_by("score")
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
        e_depth[e_i] = len(c.reg_gate_history(reg=e_i)[1])-2
        # find the max topological depth between two consecutive measurements on the same emitter
        node_list = []
        for i, oper in enumerate(c.reg_gate_history(reg=e_i)[0]):
            # first find a list of nodes in DAG corresponding to measurements
            if type(oper).__name__ in ['Input', 'MeasurementCNOTandReset', 'Output']:
                node_list.append(c.reg_gate_history(reg=e_i)[1][i])
        node_depth_list = [c._max_depth(n) for n in node_list]
        depth_diff = [node_depth_list[j+1]-node_depth_list[j] for j in range(len(node_list)-1)]
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
result["std of score"] = [np.std(x.all_scores) for x in solver.mc_list] if settings.monte_carlo else len(result) * [0]

# graph metric:
result.add_properties("graph_metric")
# a bit complicated: for each index in the result object, there is a corresponding dict of graph_metrics.
# for each graph, this dict contains the metric values for the metrics selected by user: {"met1": val1, ...} for each g
result["graph_metric"] = [dict() for _ in range(len(result))]
graph_met_list = ["node_connect"] # input
for met in graph_met_list:
    for i, g in enumerate(result["g"]):
        result["graph_metric"][i][met] = graph_met_value(met, g)

# %% Plot

met_hist(result, "score")
met_hist(result, "n_cnots")
met_hist(result, "depth")

met_met(result, "node_connect", "score")
met_met(result, "n_cnots", "score")



# %% Pre-Post processing
# metrics to measure:
"fidelity, CNOT count, max emitter depth (corrected), emitter effective depth, circuit depth (corrected)"
# separate the circuits with same relabeling map to be analyzed alone
# maps_list = list(set(result["map"]))
# # a dictionary between each relabel map and a list of indices in result corresponding to that map
# index_map_dict = dict(zip(maps_list, [[] for _ in maps_list]))
# for i in range(len(result)):
#     index_map_dict[result["map"][i]].append(i)


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
