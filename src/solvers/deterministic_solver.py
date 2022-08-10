"""
Deterministic solver which follows the paper by Li et al. This solver works for the stabilizer backend.
This solver is based on certain physical rules imposed by a platform.
One can set these rules by the set of allowed transformations.
"""

import numpy as np
import warnings


import src.backends.stabilizer.functions.height as height
import src.backends.stabilizer.functions.linalg as slinalg
from src.backends.stabilizer.state import Stabilizer
from src.backends.compiler_base import CompilerBase

import src.noise.noise_models as nm
from src.metrics import MetricBase
from src.solvers import SolverBase
from src.circuit import CircuitDAG
from src import ops


class DeterministicSolver(SolverBase):
    """
    Implements a deterministic solver
    """

    name = "deterministic"

    one_qubit_ops = list(ops.one_qubit_cliffords())

    def __init__(
        self,
        target,
        metric: MetricBase,
        compiler: CompilerBase,
        circuit: CircuitDAG = None,
        n_emitter=1,
        n_photon=1,
        noise_model_mapping=None,
        *args,
        **kwargs,
    ):

        super().__init__(target, metric, compiler, circuit, *args, **kwargs)

        self.n_emitter = n_emitter
        self.n_photon = n_photon
        self.noise_simulation = True

        if noise_model_mapping is None:
            noise_model_mapping = {}
            self.noise_simulation = False
        elif type(noise_model_mapping) is not dict:
            raise TypeError(
                f"Datatype {type(noise_model_mapping)} is not a valid noise_model_mapping. "
                f"noise_model_mapping should be a dict or None"
            )
        self.noise_model_mapping = noise_model_mapping

    def solve(self):
        """
        The main function for the solver
        # TODO: implement this function for the deterministic solver

        :return: nothing
        :rtype: None
        """

        self.compiler.noise_simulation = self.noise_simulation

        # initialize the circuit
        circuit = CircuitDAG(
            n_emitter=self.n_emitter, n_photon=self.n_photon, n_classical=1
        )

        # main loop
        target_x = self.target.tableau.stabilizer_x
        target_z = self.target.tableau.stabilizer_z
        # convert to echelon gauge
        target_x, target_z = slinalg.rref(target_x, target_z)
        for j in range(self.n_photon, 0, -1):

            height_list = height.height_func_list(target_x, target_z)
            if height_list[j] >= height_list[j - 1]:
                # add CNOT and update
                pass
            else:
                # add measurement and update
                pass

        # find transformation to bring all photons to |0> state

        # final circuit
        circuit.validate()

        compiled_state = self.compiler.compile(circuit)
        # this will pass out a density matrix object

        state_data = slinalg.partial_trace(
            compiled_state.data, keep=list(range(self.n_photon))
        )
        # evaluate the metric
        # score = self.metric.evaluate(state_data, circuit)

    def _wrap_noise(self, op, noise_model_mapping):
        """
        A helper function to consolidate noise models for OneQubitWrapper operation

        :param op: a list of operations
        :type op: list[ops.OperationBase]
        :param noise_model_mapping: a dictionary that stores the mapping between an operation
            and its associated noise model
        :type noise_model_mapping: dict
        :return: a list of noise models
        :rtype: list[nm.NoiseBase]
        """
        noise = []
        for each_op in op:
            noise.append(self._identify_noise(each_op.__name__, noise_model_mapping))
        return noise

    def _identify_noise(self, op, noise_model_mapping):
        """
        A helper function to identify the noise model for an operation

        :param op: an operation or its name
        :type op: ops.OperationBase or str
        :param noise_model_mapping: a dictionary that stores the mapping between an operation
            and its associated noise model
        :type noise_model_mapping: dict
        :return: a noise model
        :rtype: nm.NoiseBase
        """
        if type(op) != str:
            op_name = type(op).__name__
        else:
            op_name = op
        if op_name in noise_model_mapping.keys():
            return noise_model_mapping[op_name]
        else:
            return nm.NoNoise()

    def replace_photon_one_qubit_op(self, circuit):
        """
        Replace one one-qubit Clifford gate applied on a photonic qubit to another one.
        # TODO: check if this is needed after some modification

        :param circuit: a quantum circuit
        :type circuit: CircuitDAG
        :return: nothing
        :rtype: None
        """
        nodes = circuit.get_node_by_labels(["OneQubitGateWrapper", "Photonic"])

        if len(nodes) == 0:
            return
        ind = np.random.randint(len(nodes))
        node = list(nodes)[ind]

        old_op = circuit.dag.nodes[node]["op"]

        reg = old_op.register
        ind = np.random.choice(len(self.one_qubit_ops), p=self.p_dist)
        op = self.one_qubit_ops[ind]
        noise = self._wrap_noise(op, self.noise_model_mapping)
        gate = ops.OneQubitGateWrapper(op, reg_type="p", register=reg, noise=noise)
        gate.add_labels("Fixed")
        # circuit.replace_op(node, gate)
        circuit._openqasm_update(gate)
        circuit.dag.nodes[node]["op"] = gate

    def add_emitter_one_qubit_op(self, circuit, edge):
        """
        Insert a one-qubit operation at a specified edge


        :param circuit: a quantum circuit
        :type circuit: CircuitDAG
        :return: nothing
        :rtype: None
        """

        reg = circuit.dag.edges[edge]["reg"]
        label = edge[2]

        ind = np.random.choice(len(self.one_qubit_ops), p=self.e_dist)
        op = self.one_qubit_ops[ind]
        noise = self._wrap_noise(op, self.noise_model_mapping)
        gate = ops.OneQubitGateWrapper(op, reg_type="e", register=reg, noise=noise)

        circuit.insert_at(gate, [edge])

    def add_emitter_cnot(self, circuit, edge0, edge1):
        """
        Add a CNOT gate between two edges
        # TODO: check if this function is needed (after some modification)

        :param circuit: a quantum circuit
        :type circuit: CircuitDAG
        :return: nothing
        :rtype: None
        """

        gate = ops.CNOT(
            control=circuit.dag.edges[edge0]["reg"],
            control_type="e",
            target=circuit.dag.edges[edge1]["reg"],
            target_type="e",
            noise=self._identify_noise(ops.CNOT.__name__, self.noise_model_mapping),
        )

        circuit.insert_at(gate, [edge0, edge1])

    def add_measurement_cnot_and_reset(self, circuit, edge0, edge1):
        """
        Add a MeasurementCNOTandReset operation from an emitter qubit to a photonic qubit such that no consecutive
        MeasurementCNOTReset is allowed. This operation cannot be added before the photonic qubit is initialized.
        # TODO: check if this function is needed after some modification

        :param circuit: a quantum circuit
        :type circuit: CircuitDAG
        :return: nothing
        :rtype: None
        """

        gate = ops.MeasurementCNOTandReset(
            control=circuit.dag.edges[edge0]["reg"],
            control_type="e",
            target=circuit.dag.edges[edge1]["reg"],
            target_type="p",
            noise=self._identify_noise(
                ops.MeasurementCNOTandReset.__name__, self.noise_model_mapping
            ),
        )

        circuit.insert_at(gate, [edge0, edge1])

    @property
    def solver_info(self):
        """
        Return the solver setting

        :return: attributes of the solver
        :rtype: dict
        """

        def op_names(op_list):
            op_name = []
            for op_val in op_list:
                if isinstance(op_val, list):
                    op_name.append([op.__name__ for op in op_val])
                else:
                    op_name.append(op_val.__name__)
            return op_name

        return {
            "solver name": self.name,
            "seed": self.last_seed,
            "One-qubit ops": op_names(self.one_qubit_ops),
        }
