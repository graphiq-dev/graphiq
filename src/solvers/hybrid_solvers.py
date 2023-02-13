"""
Contains various hybrid solvers
"""
import networkx as nx
import numpy as np
import src.ops as ops

import src.utils.preprocessing as pre
import src.utils.circuit_comparison as comp
import src.backends.lc_equivalence_check as lc

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
from src.utils.relabel_module import iso_finder, emitter_sorted
from src.backends.state_representation_conversion import stabilizer_to_graph
from src.backends.stabilizer.functions.rep_conversion import (
    get_clifford_tableau_from_graph,
)
from src.backends.stabilizer.state import Stabilizer


class HybridEvolutionarySolver(EvolutionarySolver):
    """
    Implements a hybrid solver based on deterministic solver and rule-based evolutionary search solver.
    It takes the solution from DeterministicSolver (without noise simulation)
    as the starting point for the EvolutionarySolver.
    """

    name = "hybrid evolutionary-search"

    def __init__(
        self,
        target,
        metric: MetricBase,
        compiler: CompilerBase,
        io: IO = None,
        solver_setting=EvolutionarySearchSolverSetting(),
        noise_model_mapping=None,
        *args,
        **kwargs,
    ):
        """
        Initialize a hybrid solver based on DeterministicSolver and EvolutionarySolver

        :param target: target quantum state
        :type target: QuantumState
        :param metric: metric (cost) function to minimize
        :type metric: MetricBase
        :param compiler: compiler backend to use when simulating quantum circuits
        :type compiler: CompilerBase
        :param io: input/output object for saving logs, intermediate results, circuits, etc.
        :type io: IO
        :param n_hof: the size of the hall of fame (hof)
        :type n_hof: int
        :param selection_active: use selection in the evolutionary algorithm
        :type selection_active: bool
        :param use_adapt_probability: use adapted probability in the evolutionary algorithm
        :type use_adapt_probability: bool
        :param save_openqasm: save population, hof, or both to openQASM strings (options: None, "hof", "pop", "both")
        :type save_openqasm: str, None
        :param noise_model_mapping: a dictionary that associates each operation type to a noise model
        :type noise_model_mapping: dict
        """

        tableau = target.stabilizer.tableau
        n_photon = tableau.n_qubits
        n_emitter = DeterministicSolver.determine_n_emitters(tableau.to_stabilizer())
        super().__init__(
            target=target,
            metric=metric,
            compiler=compiler,
            circuit=None,
            io=io,
            n_emitter=n_emitter,
            n_photon=n_photon,
            solver_setting=solver_setting,
            noise_model_mapping=noise_model_mapping,
            *args,
            **kwargs,
        )

    def initialize_transformation_probabilities(self):
        """
        Sets the initial probabilities for selecting the circuit transformations.

        :return: the transformation probabilities for possible transformations
        :rtype: dict
        """
        trans_probs = {
            self.add_emitter_one_qubit_op: 1 / 4,
            self.replace_photon_one_qubit_op: 1 / 4,
            self.remove_op: 1 / 4,
            self.add_measurement_cnot_and_reset: 1 / 10,
        }

        if self.n_emitter > 1:
            trans_probs[self.add_emitter_cnot] = 1 / 4

        return self._normalize_trans_prob(trans_probs)

    def randomize_circuit(self, circuit):
        """
        Perform multiple random operations to a circuit

        :param circuit: a quantum circuit
        :type circuit: CircuitDAG
        :return: a new quantum circuit that is perturbed from the input circuit
        :rtype: CircuitDAG
        """
        randomized_circuit = circuit.copy()

        trans_probs = {
            self.add_emitter_one_qubit_op: 1 / 6,
            self.add_photon_one_qubit_op: 1 / 6,
            self.remove_op: 1 / 2,
        }

        if self.n_emitter > 1:
            trans_probs[self.add_emitter_cnot] = 1 / 6

        trans_probs = self._normalize_trans_prob(trans_probs)

        n_iter = np.random.randint(1, 10)

        for i in range(n_iter):
            transformation = np.random.choice(
                list(trans_probs.keys()), p=list(trans_probs.values())
            )
            transformation(randomized_circuit)

        return randomized_circuit

    def population_initialization(self):
        """
        Initialize population

        :return: an initial population
        :rtype: list
        """
        deterministic_solver = DeterministicSolver(
            target=self.target,
            metric=self.metric,
            compiler=self.compiler,
            noise_model_mapping=self.noise_model_mapping,
        )
        deterministic_solver.noise_simulation = False
        deterministic_solver.solve()
        self.compiler.noise_simulation = True
        _, ideal_circuit = deterministic_solver.result
        population = []
        for j in range(self.setting.n_pop):
            perturbed_circuit = self.randomize_circuit(ideal_circuit)
            population.append((np.inf, perturbed_circuit))
        return population


class HybridGraphSearchSolverSetting:
    """
    A class to store the solver setting of a HybridGraphSearchSolver

    """

    def __init__(
        self,
        base_solver_setting,
        allow_relabel=True,
        n_iso_graphs=10,
        rel_inc_thresh=0.5,
        allow_exhaustive=False,
        sort_emit=True,
        label_map=False,
        iso_thresh=5,
        allow_lc=True,
        n_lc_graphs=10,
        graph_metric=pre.graph_metric_lists[0],
        lc_method="max edge",
        verbose=False,
        save_openqasm: str = "none",
    ):
        self.allow_relabel = allow_relabel
        self.allow_lc = allow_lc
        self._n_iso_graphs = n_iso_graphs
        self._n_lc_graphs = n_lc_graphs
        self._verbose = verbose
        self._save_openqasm = save_openqasm
        self.base_solver_setting = base_solver_setting
        self.lc_method = lc_method
        self.graph_metric = graph_metric
        self.rel_inc_thresh = rel_inc_thresh
        self.allow_exhaustive = allow_exhaustive
        self.sort_emitter = sort_emit
        self.label_map = label_map
        self.iso_thresh = iso_thresh

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
        s = f"verbose = {self._verbose}\n"
        s += f"save_openqasm = {self.save_openqasm}\n"
        return s


class HybridGraphSearchSolver(SolverBase):
    def __init__(
        self,
        target,
        metric: MetricBase,
        compiler: CompilerBase,
        circuit: CircuitDAG = None,
        io: IO = None,
        graph_solver_setting=None,
        noise_model_mapping=None,
        base_solver=HybridEvolutionarySolver,
        base_solver_setting=EvolutionarySearchSolverSetting(),
        *args,
        **kwargs,
    ):
        if graph_solver_setting is None:
            graph_solver_setting = HybridGraphSearchSolverSetting(
                base_solver_setting,
                1,
                1,
                pre.graph_metric_lists[0],
            )
        super().__init__(
            target=target,
            metric=metric,
            compiler=compiler,
            circuit=circuit,
            io=io,
            solver_setting=graph_solver_setting,
            *args,
            **kwargs,
        )
        self.base_solver = base_solver
        if noise_model_mapping is None:
            noise_model_mapping = {"e": dict(), "p": dict(), "ee": dict(), "ep": dict()}
            self.noise_simulation = False
        elif type(noise_model_mapping) is not dict:
            raise TypeError(
                f"Datatype {type(noise_model_mapping)} is not a valid noise_model_mapping. "
                f"noise_model_mapping should be a dict or None"
            )
        else:
            self.noise_simulation = True

        self.noise_model_mapping = noise_model_mapping

    def solve(self):
        """

        :return:
        :rtype:
        """
        circuit_list = []
        # construct the adjacent matrix from the user-input target graph state
        target_graph = stabilizer_to_graph(
            self.target.stabilizer.tableau.stabilizer_to_labels()
        )
        n_qubits = self.target.n_qubits
        adj_matrix = nx.to_numpy_array(target_graph)

        # retrieve parameters for relabelling module and local complementation module
        n_iso = self.solver_setting.n_iso_graphs
        n_graphs = self.solver_setting.n_lc_graphs
        graph_metric = self.solver_setting.graph_metric

        if self.solver_setting.allow_relabel:

            iso_graphs = iso_finder(adj_matrix, n_iso, label_map=True)
            iso_graph_tuples = emitter_sorted(iso_graphs)
        else:
            n_emitter = DeterministicSolver.determine_n_emitters(
                self.target.stabilizer.tableau.stabilizer
            )
            (iso_graph_tuples,) = [(target_graph, n_emitter)]

        # add necessary parameters for both modules
        # TODO: postselect those graphs based on either emitter number or some other metrics
        # get the emitter number and pass to the solver
        # TODO: whether user can disable relabeling module
        for i in range(len(iso_graph_tuples)):

            if self.solver_setting.allow_lc:
                if self.solver_setting.lc_method == "max neighbor edge":
                    lc_graphs = pre.get_lc_equivalent_graph_max_neighbor_edge(
                        iso_graph_tuples[i][0], n_graphs, graph_metric
                    )
                elif self.solver_setting.lc_method == "max edge":
                    lc_graphs = pre.get_lc_equivalent_graph_max_edge(
                        iso_graph_tuples[i][0], n_graphs, graph_metric
                    )
                else:
                    raise ValueError(
                        f"The method {self.solver_setting.lc_method} is not valid."
                    )
            else:
                lc_graphs = [iso_graph_tuples[i][0]]

            # reset the target state in the metric
            tableau = get_clifford_tableau_from_graph(
                nx.from_numpy_array(iso_graph_tuples[i][0])
            )
            stabilizer = Stabilizer(tableau)
            target_state = QuantumState(n_qubits, stabilizer, "stabilizer")
            self.metric.target = target_state

            for graph in lc_graphs:
                # solve the noise-free scenario
                tableau = get_clifford_tableau_from_graph(nx.from_numpy_array(graph))
                lc_stabilizer = Stabilizer(tableau)
                solver_target_state = QuantumState(
                    n_qubits, lc_stabilizer, "stabilizer"
                )
                solver = self.base_solver(
                    target=solver_target_state,
                    metric=self.metric,
                    compiler=self.compiler,
                    n_photon=self.n_photon,
                    n_emitter=iso_graph_tuples[i][1],
                    solver_setting=self.solver_setting.base_solver_setting,
                )
                # run the solver without noise
                solver.noise_simulation = False
                solver.solve()
                score, circuit = solver.result

                # TODO: need to modify the local Clifford equivalency code to allow stabilizer comparisons
                op_list = lc.find_lc_operations(stabilizer, lc_stabilizer)
                self._add_gates_from_str(circuit, op_list)

                # store score and circuit for further analysis
                # code for storing the necessary information
                circuit_list.append(circuit)

            # code for circuit equivalency check
            comp.remove_redundant_circuits(circuit_list)

            # code to run each circuit in the noisy scenario and evaluate the cost function
            self.circuit_evaluation(circuit_list, self.metric)

    def circuit_evaluation(self, circuit_list, metric):
        # TODO: finish the implementation of this function
        self.compiler.noise_simulation = True
        score_list = []
        for circuit in circuit_list:
            compiled_state = self.compiler.compile(circuit)
            # trace out emitter qubits
            compiled_state.partial_trace(
                keep=list(range(self.n_photon)),
                dims=(self.n_photon + self.n_emitter) * [2],
            )
            # evaluate the metric
            score = metric.evaluate(compiled_state, circuit)
            score_list.append(score)

        index_list = np.argsort(score_list)
        sorted_circuit_list = [circuit_list[index] for index in index_list]
        return sorted_circuit_list

    """
    Code that seems duplicate from deterministic solver with minor modifications
    """

    def _add_gates_from_str(self, circuit, gate_str_list):
        """
        Add gates to disentangle all emitter qubits. This is used in the last step.

        :param circuit: a quantum circuit
        :type circuit: CircuitDAG
        :param gate_str_list: a list of gates to be applied
        :type gate_str_list: list[(str, int) or (str, int, int)]
        :return: nothing
        :rtype: None
        """

        for gate in gate_str_list:
            if gate[0] == "H":
                self._add_one_qubit_gate(circuit, [ops.Hadamard], gate[1])

            elif gate[0] == "P":
                # add the inverse of the phase gate
                self._add_one_qubit_gate(circuit, [ops.SigmaZ, ops.Phase], gate[1])

            elif gate[0] == "X":
                self._add_one_qubit_gate(circuit, [ops.SigmaX], gate[1])

            elif gate[0] == "CNOT":
                self._add_one_emitter_cnot(
                    circuit, gate[1] - self.n_photon, gate[2] - self.n_photon
                )

            elif gate[0] == "CZ":
                self._add_one_qubit_gate(circuit, [ops.Hadamard], gate[2])

                self._add_one_emitter_cnot(
                    circuit, gate[1] - self.n_photon, gate[2] - self.n_photon
                )

                self._add_one_qubit_gate(circuit, [ops.Hadamard], gate[2])

            else:
                raise ValueError("Invalid gate in the list.")

    def _add_one_emitter_cnot(self, circuit, control_emitter, target_emitter):
        """
        Add a CNOT between two emitter qubits

        :param circuit: a quantum circuit
        :type circuit: CircuitDAG
        :param control_emitter: register index of the control emitter
        :type control_emitter: int
        :param target_emitter: register index of the target emitter
        :type target_emitter: int
        :return: nothing
        :rtype: None
        """

        control_edge = circuit.dag.out_edges(nbunch=f"e{control_emitter}_in", keys=True)
        edge0 = list(control_edge)[0]
        target_edge = circuit.dag.out_edges(nbunch=f"e{target_emitter}_in", keys=True)
        edge1 = list(target_edge)[0]
        gate = ops.CNOT(
            control=circuit.dag.edges[edge0]["reg"],
            control_type="e",
            target=circuit.dag.edges[edge1]["reg"],
            target_type="e",
            noise=self._identify_noise(ops.CNOT, self.noise_model_mapping["ee"]),
        )
        circuit.insert_at(gate, [edge0, edge1])

    def _add_one_qubit_gate(self, circuit, gate_list, index):
        """
        Add a one-qubit gate to the circuit

        :param circuit: a quantum circuit
        :type circuit: CircuitDAG
        :param gate_list: a list of one-qubit gates to be added to the circuit
        :type gate_list: list[ops.OperationBase]
        :param index: the qubit position where this one-qubit gate is applied
        :type index: int
        :return: nothing
        :rtype: None
        """

        if index >= self.n_photon:
            reg_type = "e"
            reg = index - self.n_photon
        else:
            reg_type = "p"
            reg = index

        edge = circuit.dag.out_edges(nbunch=f"{reg_type}{reg}_in", keys=True)
        edge = list(edge)[0]
        next_node = circuit.dag.nodes[edge[1]]
        next_op = next_node["op"]
        if isinstance(next_op, ops.OneQubitGateWrapper):
            # if two OneQubitGateWrapper gates are next to each other, combine them
            gate_list = next_op.operations + gate_list
            gate_list = ops.simplify_local_clifford(gate_list)
            if gate_list == [ops.Identity, ops.Identity]:
                circuit.remove_op(edge[1])
                return
        else:
            # simplify the gate to be one of the 24 local Clifford gates
            gate_list = ops.simplify_local_clifford(gate_list)
            if gate_list == [ops.Identity, ops.Identity]:
                return
        if reg_type == "e":
            noise = self._wrap_noise(gate_list, self.noise_model_mapping["e"])
        else:
            noise = self._wrap_noise(gate_list, self.noise_model_mapping["p"])
        gate = ops.OneQubitGateWrapper(
            gate_list,
            reg_type=reg_type,
            register=reg,
            noise=noise,
        )
        if isinstance(next_op, ops.OneQubitGateWrapper):
            circuit.replace_op(edge[1], gate)
        else:
            circuit.insert_at(gate, [edge])
