"""
AlternateTargetSolver uses alternative target graph states as the starting point for the TimeReversedSolver.
Alternative target graph states can be found by relabeling (photon emission ordering)
and local complementation (local Clifford equivalency).
"""

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from graphiq.benchmarks.graph_states import repeater_graph_states
from graphiq.backends.compiler_base import CompilerBase
from graphiq.backends.density_matrix.compiler import DensityMatrixCompiler
from graphiq.backends.stabilizer.compiler import StabilizerCompiler
from graphiq.backends.stabilizer.functions import local_cliff_equi_check as slc
from graphiq.backends.stabilizer.functions.rep_conversion import (
    get_clifford_tableau_from_graph,
)
from graphiq.backends.state_rep_conversion import (
    stabilizer_to_graph,
    graph_to_density,
    density_to_graph,
)
from graphiq.io import IO
from graphiq.metrics import MetricBase, Infidelity
from graphiq.noise import monte_carlo_noise as mcn, noise_models as nm
from graphiq.solvers.time_reversed_solver import TimeReversedSolver
from graphiq.solvers.solver_result import SolverResult
from graphiq.state import QuantumState
from graphiq.utils import preprocessing as pre
from graphiq.utils.relabel_module import (
    iso_finder,
    rgs_orbit_finder,
    linear_partial_orbit,
    depth_first_orbit,
    lc_orbit_finder,
    get_relabel_map,
    relabel,
)


class AlternateTargetSolver:
    def __init__(
        self,
        target: nx.Graph or QuantumState = None,
        metric: MetricBase = Infidelity,
        compiler: CompilerBase = StabilizerCompiler(),
        noise_compiler: CompilerBase = DensityMatrixCompiler(),
        io: IO = None,
        noise_model_mapping=None,
        solver_setting=None,
        seed=None,
    ):
        """
        Constructor for AlternateGraphSolver

        :param target: target graph state
        :type target: QuantumState or nx.Graph
        :param metric: a metric
        :type metric: MetricBase
        :param compiler: a compiler
        :type compiler: CompilerBase
        :param noise_compiler: compiler for noise simulation
        :type noise_compiler: CompilerBase
        :param io: input/output
        :type io: IO
        :param noise_model_mapping:
        :type noise_model_mapping: dict
        :param solver_setting:
        :type solver_setting:
        :param seed: a random seed
        :type seed: int
        """
        if solver_setting is None:
            solver_setting = AlternateTargetSolverSetting(
                n_iso_graphs=1, n_lc_graphs=1, graph_metric=pre.graph_metric_lists[0]
            )
        if noise_model_mapping is None:
            noise_model_mapping = {"e": dict(), "p": dict(), "ee": dict(), "ep": dict()}
            self.noise_simulation = False
            self.monte_carlo = False
        elif noise_model_mapping == "depolarizing":
            self.noise_simulation = True
            self.depolarizing_rate = solver_setting.depolarizing_rate
            if solver_setting.monte_carlo:
                self.monte_carlo = True
                self.mc_params = solver_setting.monte_carlo_params
                self.mc_params["map"] = self.mc_depol()
            else:
                self.monte_carlo = False
                noise_model_mapping = self.depol_noise_map()
        elif type(noise_model_mapping) is not dict:
            raise TypeError(
                f"Datatype {type(noise_model_mapping)} is not a valid noise_model_mapping. "
                f"noise_model_mapping should be a dict or None"
            )
        else:
            self.noise_simulation = True
            self.monte_carlo = False
            if solver_setting.monte_carlo:
                self.monte_carlo = True
                self.mc_params = solver_setting.monte_carlo_params

        self.noise_model_mapping = noise_model_mapping

        if isinstance(target, nx.Graph):
            self.target_graph = target
            self.target = QuantumState(target, rep_type="g")
        elif isinstance(target, QuantumState):
            if target.mixed:
                raise ValueError(
                    "AlternateTargetSolver does not support mixed states as its target."
                )
            self.target = target
            if target.rep_type == "s":
                target_graph = stabilizer_to_graph(
                    target.rep_data.tableau.to_stabilizer()
                )
                self.target_graph = target_graph[0][1]
            elif target.rep_type == "dm":
                target_graph = density_to_graph(target.rep_data.data)
                self.target_graph = nx.from_numpy_array(target_graph)

            elif target.rep_type == "g":
                self.target_graph = target.rep_data.data
            else:
                raise ValueError("Wrong representation data type for QuantumState")

        self.metric = metric
        self.noise_compiler = noise_compiler
        self.compiler = compiler
        self.solver_setting = solver_setting
        self.seed = seed
        self.result = None
        self.mc_list = []

    def solve(self):
        """
        Finds alternative circuits to generate the target graph or a relabeled version of it by searching through
        alternative isomorphic and LC-equivalent graphs. Notice that the returned circuits generate the target state
        up to relabeling if this feature is enabled in the setting. Otherwise, user's exact target graph is produced.

        :return: a dictionary where keys are the circuits and values are themselves dictionaries containing the LC graph
         used as the intermediate states and relabeling map between the actual target and the graph the circuit
         generates. {circuit: {'g':graph, 'map': relabel_map}}
        :rtype: dict
        """
        setting = self.solver_setting
        n_iso = setting.n_iso_graphs
        n_lc = setting.n_lc_graphs
        adj_matrix = nx.to_numpy_array(self.target_graph)

        iso_adjs = iso_finder(
            adj_matrix,
            n_iso,
            rel_inc_thresh=setting.rel_inc_thresh,
            allow_exhaustive=setting.allow_exhaustive,
            sort_emit=setting.sort_emitter,
            label_map=setting.label_map,
            thresh=setting.iso_thresh,
            seed=self.seed,
        )
        results_list = []
        mc_list = []
        # repeater graph state check
        adj_rgs = nx.to_numpy_array(
            repeater_graph_states(int(len(self.target_graph) / 2))
        )
        target_is_rgs = True if np.array_equal(adj_rgs, adj_matrix) else False
        # list of tuples [(circuit, dict={'g': graph used to find circuit, 'map': relabel map with target})]
        iso_graphs = [nx.from_numpy_array(adj) for adj in iso_adjs]
        for iso_graph in iso_graphs:
            if setting.lc_method == "rgs":
                lc_graphs = rgs_orbit_finder(iso_graph)[:n_lc]
            elif setting.lc_method == "linear":
                lc_graphs = linear_partial_orbit(iso_graph)[:n_lc]
            elif setting.lc_method == "depth_first":
                lc_graphs = depth_first_orbit(iso_graph)[:n_lc]
            elif setting.lc_method == "lc_with_iso":
                lc_graphs = lc_orbit_finder(
                    iso_graph,
                    comp_depth=setting.lc_orbit_depth,
                    orbit_size_thresh=n_lc,
                    with_iso=True,
                )
            elif setting.lc_method == "random":
                lc_graphs = lc_orbit_finder(
                    iso_graph,
                    comp_depth=setting.lc_orbit_depth,
                    orbit_size_thresh=n_lc,
                    rand=True,
                )
            elif setting.lc_method == "random_with_iso":
                lc_graphs = lc_orbit_finder(
                    iso_graph,
                    comp_depth=setting.lc_orbit_depth,
                    orbit_size_thresh=n_lc,
                    with_iso=True,
                    rand=True,
                )
            elif setting.lc_method == "random_with_rep":
                lc_graphs = lc_orbit_finder(
                    iso_graph,
                    comp_depth=setting.lc_orbit_depth,
                    orbit_size_thresh=n_lc,
                    with_iso=True,
                    rand=True,
                    rep_allowed=True,
                )
            elif setting.lc_method is None:
                lc_graphs = lc_orbit_finder(
                    iso_graph, comp_depth=setting.lc_orbit_depth, orbit_size_thresh=n_lc
                )
            else:
                raise ValueError(
                    "LC method is not valid. Please set it to a valid string or None"
                )
            lc_circ_list = []
            lc_score_list = []
            rmap = get_relabel_map(self.target_graph, iso_graph)

            for lc_graph in lc_graphs:
                circuit = graph_to_circ(lc_graph)
                success, conversion_gates = slc.lc_check(
                    lc_graph, iso_graph, validate=True
                )
                try:
                    assert success, "LC graphs are not actually LC equivalent!"
                    slc.state_converter_circuit(lc_graph, iso_graph, validate=True)
                except:
                    raise UserWarning("LC conversion failed")
                conversion_ops = slc.str_to_op(conversion_gates)
                for op in conversion_ops:
                    circuit.add(op)
                if self.noise_simulation:
                    if self.monte_carlo:
                        # generate a unique seed for the monte carlo object based on user input seeds
                        if self.seed is not None and self.mc_params["seed"] is not None:
                            rng = np.random.default_rng(
                                self.seed * self.mc_params["seed"]
                            )
                        else:
                            # seedless randomness if at least one of the user's seeds is None
                            rng = np.random.default_rng()
                        n_mc = max(
                            int(self.mc_params["n_sample"] or 0),
                            int(self.mc_params["n_single"] or 0)
                            * int(self.mc_params["n_parallel"] or 0),
                        )
                        mc_seed = rng.integers(
                            low=1, high=int(10e6 * max(n_mc, 2)), size=1
                        )
                        # end of random seed generation
                        mc = mcn.MonteCarloNoise(
                            circuit,
                            self.mc_params["n_sample"],
                            self.mc_params["map"],
                            self.mc_params["compiler"],
                            mc_seed,
                        )
                        mc_list.append(mc)
                        if self.mc_params["n_parallel"] is not None:
                            n_total = (
                                self.mc_params["n_parallel"]
                                * self.mc_params["n_single"]
                            )
                            assert (
                                n_total > 0
                            ), "n_single and n_parallel both must be integers > 1 or None"
                            self.solver_setting.monte_carlo_params["n_sample"] = n_total
                            # multicore parallel processing
                            # ray.init()
                            noise_score = mcn.parallel_monte_carlo(
                                mc,
                                self.mc_params["n_parallel"],
                                self.mc_params["n_single"],
                            )
                            # ray.shutdown()

                        else:
                            noise_score = mc.run()
                    else:
                        circuit = circuit.assign_noise(self.noise_model_mapping)
                        noise_score = self.noise_score(circuit, rmap)
                    lc_score_list.append(noise_score)
                else:
                    lc_score_list.append(0.0001)

                lc_circ_list.append(circuit)

            for i, circ in enumerate(lc_circ_list):
                results_list.append(
                    (circ, {"g": lc_graphs[i], "score": lc_score_list[i], "map": rmap})
                )
            # iso_lc_circuit_dict[get_relabel_map(self.target, iso_graph)] = dict(zip(lc_graphs, lc_circ_list))
        # remove redundant auto-morph graphs
        adj_list = [nx.to_numpy_array(result[1]["g"]) for result in results_list]
        set_list = []
        for i in range(len(adj_list)):
            already_found = False
            for s in set_list:
                if i in s:
                    already_found = True
                    break
            if not already_found:
                s = {i}
                for j in range(i + 1, len(adj_list)):
                    if np.array_equal(adj_list[i], adj_list[j]):
                        s.add(j)
                set_list.append(s)
        # set_list now contains the equivalent group of graphs in the result's dict
        # remove redundant items of the dict
        redundant_indices = [index for s in set_list for index in list(s)[1:]]
        redundant_indices.sort()
        for index in redundant_indices[::-1]:
            del results_list[index]

        # results setter
        self.mc_list = mc_list
        circ_list = [x[0] for x in results_list]
        properties = list(results_list[0][1].keys()) if results_list else None
        self.result = SolverResult(circ_list, properties)
        # [r[1][p] for p in properties for r in results_list]
        for p in properties:
            self.result[p] = [r[1][p] for r in results_list]

        return results_list

    def noise_score(self, circuit, relabel_map):
        """


        :param circuit:
        :type circuit:
        :param relabel_map:
        :type relabel_map:
        :return:
        :rtype:
        """
        graph = self.target_graph
        old_adj = nx.to_numpy_array(graph)
        n = graph.number_of_nodes()
        # relabel the target graph
        new_labels = np.array([relabel_map[i] for i in range(n)])
        new_adj = relabel(old_adj, new_labels)
        graph = nx.from_numpy_array(new_adj)

        c_tableau = get_clifford_tableau_from_graph(graph)
        ideal_state = QuantumState(c_tableau, rep_type="s")
        if isinstance(self.noise_compiler, DensityMatrixCompiler):
            ideal_state = QuantumState(graph_to_density(graph), rep_type="dm")
        metric = Infidelity(ideal_state)

        compiler = self.noise_compiler
        compiler.noise_simulation = True
        compiled_state = compiler.compile(circuit=circuit)
        # trace out emitter qubits
        compiled_state.partial_trace(
            keep=list(range(circuit.n_photons)),
            dims=(circuit.n_photons + circuit.n_emitters) * [2],
        )

        # evaluate the metric
        score = metric.evaluate(compiled_state, circuit)
        score = 0.00001 if (np.isclose(score, 0.0) and score != 0.0) else score
        return score

    def depol_noise_map(self):
        """


        :return:
        :rtype:
        """

        rate = self.depolarizing_rate
        dep_noise_model_mapping = dict()
        dep_noise_model_mapping["e"] = {
            # "SigmaX": nm.DepolarizingNoise(rate),
            # "SigmaY": nm.DepolarizingNoise(rate),
            # "SigmaZ": nm.DepolarizingNoise(rate),
            "Phase": nm.DepolarizingNoise(rate),
            "PhaseDagger": nm.DepolarizingNoise(rate),
            "Hadamard": nm.DepolarizingNoise(rate),
        }
        dep_noise_model_mapping["p"] = {}  # dep_noise_model_mapping["e"]
        # dep_noise_model_mapping["ee"] = {}
        # dep_noise_model_mapping["ep"] = {}
        dep_noise_model_mapping["ee"] = {"CNOT": nm.DepolarizingNoise(rate)}
        dep_noise_model_mapping["ep"] = {"CNOT": nm.DepolarizingNoise(rate)}
        return dep_noise_model_mapping

    def mc_depol(self):
        """
        Returns a Monte-Carlo noise map for depolarizing noise. Currently only emitter gates are noisy.

        :return: mcn.McNoiseMap
        :rtype: mcn.McNoiseMap
        """
        rate = self.depolarizing_rate / 3
        mc_noise = mcn.McNoiseMap()
        mc_noise.add_gate_noise(
            "e",
            "Hadamard",
            [
                (nm.PauliError("X"), rate),
                (nm.PauliError("Y"), rate),
                (nm.PauliError("Z"), rate),
            ],
        )
        mc_noise.add_gate_noise(
            "e",
            "Phase",
            [
                (nm.PauliError("X"), rate),
                (nm.PauliError("Y"), rate),
                (nm.PauliError("Z"), rate),
            ],
        )
        mc_noise.add_gate_noise(
            "e",
            "PhaseDagger",
            [
                (nm.PauliError("X"), rate),
                (nm.PauliError("Y"), rate),
                (nm.PauliError("Z"), rate),
            ],
        )
        mc_noise.add_gate_noise(
            "ee",
            "CNOT",
            [
                (nm.PauliError("X"), rate),
                (nm.PauliError("Y"), rate),
                (nm.PauliError("Z"), rate),
            ],
        )
        mc_noise.add_gate_noise(
            "ep",
            "CNOT",
            [
                (nm.PauliError("X"), rate),
                (nm.PauliError("Y"), rate),
                (nm.PauliError("Z"), rate),
            ],
        )
        return mc_noise


def graph_to_circ(graph, noise_model_mapping=None, show=False):
    """
    Find a circuit that generates the input graph. This function calls the deterministic solver. The outcome is not
    unique.

    :param graph: The graph to generate
    :type graph: networkx.Graph
    :param noise_model_mapping:
    :type noise_model_mapping: dict
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
    ideal_state = QuantumState(c_tableau, rep_type="s")

    target = ideal_state
    solver = TimeReversedSolver(
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


class AlternateTargetSolverSetting:
    """
    A class to store the solver setting of an AlternateTargetSolver

    """

    def __init__(
        self,
        allow_relabel=True,
        n_iso_graphs=10,
        rel_inc_thresh=0.1,
        allow_exhaustive=False,
        sort_emit=True,
        label_map=False,
        iso_thresh=None,
        allow_lc=True,
        n_lc_graphs=10,
        lc_orbit_depth=None,
        depolarizing_rate=0.01,
        monte_carlo=False,
        monte_carlo_params=None,
        graph_metric=pre.graph_metric_lists[0],
        lc_method="max edge",
        verbose=False,
        save_openqasm: str = "none",
        callback_func: dict = {},
    ):
        self.allow_relabel = allow_relabel
        self.allow_lc = allow_lc
        self._n_iso_graphs = n_iso_graphs
        self._n_lc_graphs = n_lc_graphs
        self._verbose = verbose
        self._save_openqasm = save_openqasm
        self.lc_method = lc_method
        self.graph_metric = graph_metric
        self.rel_inc_thresh = rel_inc_thresh
        self.allow_exhaustive = allow_exhaustive
        self.sort_emitter = sort_emit
        self.label_map = label_map
        self.iso_thresh = iso_thresh
        self.callback_func = callback_func
        self.lc_orbit_depth = lc_orbit_depth
        self.depolarizing_rate = depolarizing_rate
        self.monte_carlo = monte_carlo
        mc_params = {
            "n_sample": 1,
            "map": mcn.McNoiseMap(),
            "compiler": None,
            "seed": None,
            "n_parallel": None,
            "n_single": None,
        }
        if monte_carlo_params is None:
            monte_carlo_params = mc_params
        else:
            monte_carlo_params = mc_params | monte_carlo_params
        self.monte_carlo_params = monte_carlo_params

    @property
    def n_iso_graphs(self):
        return self._n_iso_graphs

    @n_iso_graphs.setter
    def n_iso_graphs(self, value):
        assert type(value) == int
        self._n_iso_graphs = value

    @property
    def n_lc_graphs(self):
        return self._n_lc_graphs

    @n_lc_graphs.setter
    def n_lc_graphs(self, value):
        assert type(value) == int or value is None
        self._n_lc_graphs = value

    @property
    def verbose(self):
        return self._verbose

    @verbose.setter
    def verbose(self, value):
        assert type(value) == bool
        self._verbose = value

    @property
    def save_openqasm(self):
        return self._save_openqasm

    @save_openqasm.setter
    def save_openqasm(self, value):
        assert type(value) == str
        self._save_openqasm = value

    def __str__(self):
        s = f"n_iso_graphs = {self._n_iso_graphs}\n"
        s += f"n_lc_graphs = {self._n_lc_graphs}\n"
        s += f"verbose = {self._verbose}\n"
        s += f"save_openqasm = {self.save_openqasm}\n"
        return s
