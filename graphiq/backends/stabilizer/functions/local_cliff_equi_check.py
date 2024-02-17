# Copyright (c) 2022-2024 Quantum Bridge Technologies Inc.
# Copyright (c) 2022-2024 Ki3 Photonics Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import networkx as nx
import numpy as np

import graphiq.circuit.ops as ops
from graphiq.backends.lc_equivalence_check import is_lc_equivalent, local_clifford_ops
from graphiq.backends.stabilizer.clifford_tableau import CliffordTableau
from graphiq.backends.stabilizer.compiler import StabilizerCompiler
from graphiq.backends.stabilizer.functions.rep_conversion import (
    get_stabilizer_tableau_from_graph,
    clifford_from_stabilizer,
)
from graphiq.backends.stabilizer.functions.stabilizer import canonical_form
from graphiq.backends.stabilizer.functions.transformation import run_circuit
from graphiq.backends.stabilizer.tableau import StabilizerTableau
from graphiq.backends.state_rep_conversion import (
    state_to_graph,
    _phase_correction,
)
from graphiq.circuit.circuit_dag import CircuitDAG
from graphiq.metrics import Infidelity
from graphiq.state import QuantumState


def lc_check(state1, state2, validate=True):
    """
    Takes two quantum states (or Stabilizer/Clifford tableau, graph, adjacency matrix) and checks the LC
    equivalence between them. If True, a sequence of gates to convert state1 to state 2 is also returned to.

    :param state1: the first state.
    :type state1: StabilizerTableau or CliffordTableau or nx.Graph or np.ndarray
    :param state2: the second state.
    :type state2: StabilizerTableau or CliffordTableau or nx.Graph or np.ndarray
    :param validate: if True, the provided list of gates is run on the state1 to confirm it ends up on state2 after the
    sequence.
    :type validate: bool
    :return: tuple (result, gate_list)
    :rtype: tuple (bool, list)
    """
    # determines whether two states 1 and 2 are LC equivalent;
    # if yes, determine the gate sequence that transforms the state 1 into state 2;
    # if validate is True, confirm that after applying gates to state 1, it becomes state 2.
    graph1, tab1, gates1 = state_to_graph(state1)
    graph2, tab2, gates2 = state_to_graph(state2)

    try:
        gate_list = converter_gate_list(graph1, graph2)
    except:
        return False, []
    # invert the sequence of gates that was used to convert state2 into graph2. Since we are converting state1 to state2
    # in the total gate list we must convert the graph2 to state 2 in the last step, hence the reversed gate list

    # replace P_dag operations with P in gates2, since they are inverse of each other. Other gates are self inverse
    # so no need to replace them. (and there is only X and P_dag gates in the gates1 and gates2 lists)
    inversed_gates2 = []
    for gate in gates2:
        if gate[0] == "P_dag":
            inversed_gates2.append(("P", gate[1]))
        elif gate[0] == "P":
            inversed_gates2.append(("P_dag", gate[1]))
        else:
            inversed_gates2.append(gate)
    # reverse the order too
    inversed_gates2 = inversed_gates2[::-1]

    total_gate_list = gates1 + gate_list + inversed_gates2

    # validate
    if validate:
        # check if new_tab is the same as the stabilizer tableau for state2
        tab2_can = canonical_form(tab2.copy())
        final_tab = run_circuit(tab1.copy(), total_gate_list)
        final_tab_can = canonical_form(final_tab)

        if not (tab2_can == final_tab_can):  # z_equal and x_equal and phase_equal
            raise Warning(
                "the gate sequence is not converting the state1's stabilizer tableau into state2's"
            )

    return True, total_gate_list


def state_converter_circuit(state1, state2, validate=False):
    """
    This function returns a piece of circuit that converts state1 to state2; if they are local clifford equivalent.
    The circuit corresponds to the gate_list which is returned by the function 'lc_check'.

    :param state1: the initial state
    :type state1: StabilizerTableau or CliffordTableau or nx.Graph or np.ndarray
    :param state2: the target state
    :type state2: StabilizerTableau or CliffordTableau or nx.Graph or np.ndarray
    :param validate: if True, validates the circuit to see if final output of the circuit is equal to the desired state.
    :type validate: bool
    :return: the circuit to convert graph1 to graph2
    :rtype: CircuitDAG
    """

    graph1, tab1, gates1 = state_to_graph(state1)
    graph2, tab2, gates2 = state_to_graph(state2)

    target1_tableau = clifford_from_stabilizer(tab1)
    target2_tableau = clifford_from_stabilizer(tab2)
    n_photon = target1_tableau.n_qubits

    lc, gate_list = lc_check(state1, state2)
    assert lc, "the two graphs are not LC equivalent!"

    circuit = CircuitDAG(n_photon=n_photon)
    for gate in gate_list:
        op = str_to_op(gate)
        circuit.add(op)
    if validate:
        target1 = QuantumState(target1_tableau, rep_type="stab")
        target2 = QuantumState(target2_tableau, rep_type="stab")
        metric2 = Infidelity(target2)
        compiler = StabilizerCompiler()
        final_state = compiler.compile(circuit, initial_state=target1)
        final_score = metric2.evaluate(final_state, circuit)
        assert np.isclose(
            final_score, 0
        ), "the final compiled state is not equal to the desired target state"

    return circuit


def converter_gate_list(g1, g2):
    """
    Given two graphs g1 and g2, this functions returns a sequence of gates needed to convert g1 into g2 if they are LC
    equivalent.

    :param g1: first graph
    :type g1: nx.Graph
    :param g2: second graph
    :type g2: nx.Graph
    :return: list of gates, which are (operation, qubit) tuple
    :rtype: list
    """
    adj_g1 = nx.to_numpy_array(g1)
    adj_g2 = nx.to_numpy_array(g2)
    lc, sol = is_lc_equivalent(adj_g1, adj_g2)
    assert lc, "the two graphs are not LC equivalent"
    lc_ops = local_clifford_ops(sol)
    gate_list = []
    for i, ops in enumerate(lc_ops):
        for op in ops.split()[::-1]:
            gate_list.append((op, i))
    tab1 = get_stabilizer_tableau_from_graph(g1)
    tab2 = get_stabilizer_tableau_from_graph(g2)
    phase_correction = _phase_correction(tab1, tab2, gate_list)
    gate_list += phase_correction
    return gate_list


def str_to_op(gate_tuples):
    """
    Converts a gate list, made up of gate tuples ("gate name", qubit index) used in the stabilizer backend, into a list
    of operations ready to be added to a circuit. Qubits are photons by default, not emitters.

    :param gate_tuples: a gate tuple or list of tuples
    :type gate_tuples: tuple or list
    :return: an operation or list of operations
    :rtype: ops.OneQubitOperationBase or list
    """
    # converts the (single qubit gate name, qubit index) into an OneQubitOperationBase object.
    # input gate is a tuple (name, qubit index) or a list of them
    name_list = ["I", "H", "X", "P", "P_dag", "Z"]
    ops_list = [
        ops.Identity,
        ops.Hadamard,
        ops.SigmaX,
        ops.Phase,
        ops.PhaseDagger,
        ops.SigmaZ,
    ]
    if isinstance(gate_tuples, tuple) and len(gate_tuples) == 2:
        op_index = name_list.index(gate_tuples[0])
        operation = ops_list[op_index](register=gate_tuples[1], reg_type="p")
        return operation
    elif isinstance(gate_tuples, list):
        operations_list = []
        for gate in gate_tuples:
            op_index = name_list.index(gate[0])
            operations_list.append(ops_list[op_index](register=gate[1], reg_type="p"))
        return operations_list
    else:
        raise ValueError(
            "input should be gate tuples ('gate name', qubit index) or a list of them"
        )
