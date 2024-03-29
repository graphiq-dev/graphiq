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
"""
Functions to compare quantum circuits

"""

# TODO: GED implementation requires further investigation.
import networkx as nx
from networkx.algorithms.isomorphism import is_isomorphic

from graphiq.circuit.ops import *


def compare_circuits(circuit1, circuit2, method="direct"):
    """
    Compare two circuits by using GED or direct loop method

    :param circuit1: circuit that to be compared
    :type circuit1: CircuitDAG
    :param circuit2: circuit that to be compared
    :type circuit2: CircuitDAG
    :param method: Determine which comparison function to use
    :type method: str
    :return: whether two circuits are the same
    :rtype: bool
    """
    if method == "direct":
        return direct(circuit1, circuit2)
    elif method == "GED_full":
        return ged(circuit1, circuit2, full=True)
    elif method == "GED_approximate":
        return ged(circuit1, circuit2, full=False)
    elif method == "GED_adaptive":
        return ged_adaptive(circuit1, circuit2)
    elif method == "is_isomorphic":
        return circuit_is_isomorphic(circuit1, circuit2)
    else:
        raise ValueError(f"Method {method} is not supported.")


def direct(circuit1, circuit2):
    """
    Directly compare two circuits by iterating from input nodes to output nodes

    :param circuit1: circuit that to be compared
    :type circuit1: CircuitDAG
    :param circuit2: circuit that to be compared
    :type circuit2: CircuitDAG
    :return: whether two circuits are the same
    :rtype: bool
    """
    circuit1 = circuit1.copy()
    circuit1.unwrap_nodes()
    circuit2 = circuit2.copy()
    circuit2.unwrap_nodes()

    circuit1.remove_identity()
    circuit2.remove_identity()

    n_reg_match = circuit1.register == circuit2.register
    n_nodes_match = circuit1.dag.number_of_nodes() == circuit2.dag.number_of_nodes()

    if n_reg_match and n_nodes_match:
        for in_node in circuit1.node_dict["Input"]:
            node1 = in_node
            node2 = in_node
            op = circuit1.dag.nodes[node1]["op"]
            reg = f"{op.reg_type}{op.register}"
            out_node = f"{reg}_out"
            while node1 != out_node:
                out_edge = [
                    edge
                    for edge in list(circuit1.dag.out_edges(node1, keys=True))
                    if edge[2] == reg
                ]
                node1 = out_edge[0][1]

                out_edge_compare = [
                    edge
                    for edge in list(circuit2.dag.out_edges(node2, keys=True))
                    if edge[2] == reg
                ]
                node2 = out_edge_compare[0][1]

                op1 = circuit1.dag.nodes[node1]["op"]
                op2 = circuit2.dag.nodes[node2]["op"]
                control_match = (
                    op1.q_registers_type == op2.q_registers_type
                    and op1.q_registers == op2.q_registers
                )
                if isinstance(op1, type(op2)) and control_match:
                    pass
                else:
                    return False
        return True
    else:
        return False


def ged_adaptive(circuit1, circuit2, threshold=30):
    """
    Switch between exact and approximate GED calculation adaptively

    :param circuit1: circuit that to be compared
    :type circuit1: CircuitDAG
    :param circuit2: circuit that to be compared
    :type circuit2: CircuitDAG
    :param threshold: threshold
    :type threshold: int
    :return: exact/approximated GED between circuits(cost needed to transform self.dag to circuit_compare.dag)
    :rtype: bool
    """

    full = (
        max(circuit1.dag.number_of_nodes(), circuit2.dag.number_of_nodes()) < threshold
    )
    sim = ged(circuit1, circuit2, full=full)
    return sim


def ged(circuit1, circuit2, full=True):
    """
    Calculate Graph Edit Distance (GED) between two circuits.
    Further reading on GED:
    https://networkx.org/documentation/stable/reference/algorithms/similarity.html

    :param circuit1: circuit that to be compared
    :type circuit1: CircuitDAG
    :param circuit2: circuit that to be compared
    :type circuit2: CircuitDAG
    :param full: Determine which GED function to use
    :type full: bool
    :return: whether two circuits are the same
    :rtype: bool
    """
    circuit1 = circuit1.copy()
    circuit1.unwrap_nodes()
    circuit2 = circuit2.copy()
    circuit2.unwrap_nodes()

    circuit1.remove_identity()
    circuit2.remove_identity()
    dag1 = circuit1.dag
    dag2 = circuit2.dag

    def node_match(n1, n2):
        reg_match = (
            n1["op"].q_registers_type == n2["op"].q_registers_type
            and n1["op"].q_registers == n2["op"].q_registers
        )
        ops_match = isinstance(n1["op"], type(n2["op"]))

        return reg_match and ops_match

    def edge_match(e1, e2):
        # TODO: may treat different emitters the same
        #       if the difference of two circuits is just relabeling of emitter registers
        return e1 == e2

    if full:
        sim = nx.algorithms.similarity.graph_edit_distance(
            dag1,
            dag2,
            node_match=node_match,
            edge_match=edge_match,
            upper_bound=30,
            timeout=10.0,
        )
    else:
        sim = nx.algorithms.similarity.optimize_graph_edit_distance(
            dag1,
            dag2,
            node_match=node_match,
            edge_match=edge_match,
            upper_bound=30,
        )
        sim = next(sim)

    return sim == 0


def circuit_is_isomorphic(circuit1, circuit2):
    """
    Compare 2 circuits using nx.is_isomorphic from their DAG.

    :param circuit1: circuit 1
    :type circuit1: CircuitDAG
    :param circuit2: circuit 2
    :type circuit2: CircuitDAG
    :return: True if 2 circuits is isomorphic, False otherwise.
    :rtype: Boolean
    """
    add_control_target_to_dag(circuit1)
    add_control_target_to_dag(circuit2)

    def node_match(n1, n2):
        # get operation for node 1 and node 2
        op1 = n1["op"]
        op2 = n2["op"]

        # Compare the type of the 2 operations and the q_register_type tuple
        if type(op1) != type(op2) or op1.q_registers_type != op2.q_registers_type:
            return False

        # For ControlledPairOperationBase, compare the control_type and target_type
        if isinstance(op1, ControlledPairOperationBase):
            if (
                op1.control_type != op2.control_type
                or op1.target_type != op2.target_type
            ):
                return False

        # For OneQubitGateWrapper, compare the operations list
        if type(op1) == type(op2) == OneQubitGateWrapper:
            if op1.operations != op2.operations:
                return False

        return True

    def edge_match(e1, e2):
        # Get the first key of the edge dict, normally only 1 key per edge unless we have 2 nodes that are connected by
        #  2 edges
        val1 = next(iter(e1))
        val2 = next(iter(e2))

        # Check for the control_target attribute
        if e1[val1]["control_target"] != e2[val2]["control_target"]:
            return False
        return True

    return is_isomorphic(
        circuit1.dag, circuit2.dag, node_match=node_match, edge_match=edge_match
    )


def _create_edge_control_target_attr(operation, reg_type, reg):
    """
    Helper function that return the correct control_target attribute for the edge, base on reg_type, reg, and operation.
    If the operation is ControlledPairOperationBase return either 'c' or 't' else return None.

    :param operation: operation
    :type operation: OperationBase
    :param reg_type: register type
    :type reg_type: str
    :param reg: register
    :type reg: int
    :return: control_target attribute. Can be 'c', 't' or None
    :rtype: str or nothing
    """
    if isinstance(operation, ControlledPairOperationBase):
        if reg_type == operation.control_type and reg == operation.control:
            return "c"
        if reg_type == operation.target_type and reg == operation.target:
            return "t"


def add_control_target_to_dag(circuit):
    """
    Process the input circuit DAG and add control_target attribute to edges.

    :param circuit: circuit
    :type circuit: CircuitDAG
    :return: nothing
    :rtype: None
    """

    for node in circuit.node_dict["Input"]:
        op = circuit.dag.nodes[node]["op"]
        reg_type = op.reg_type
        register = op.register

        out_edges = circuit.dag.out_edges(nbunch=node, keys=True)
        edge = circuit.edge_from_reg(out_edges, f"{reg_type}{register}")
        next_node = edge[1]
        label = edge[2]

        while next_node not in circuit.node_dict["Output"]:
            op = circuit.dag.nodes[next_node]["op"]
            control_target = _create_edge_control_target_attr(op, reg_type, register)
            circuit.dag[node][next_node][label]["control_target"] = control_target

            node = next_node
            out_edges = circuit.dag.out_edges(nbunch=node, keys=True)
            edge = circuit.edge_from_reg(out_edges, f"{reg_type}{register}")
            next_node = edge[1]
            label = edge[2]

        control_target = _create_edge_control_target_attr(op, reg_type, register)
        circuit.dag[node][next_node][label]["control_target"] = control_target


def remove_redundant_circuits(circuit_list):
    """
    The function will remove redundant circuit in a list of circuits by running the circuit_is_isomorphic() function.
    The function returns a new list of circuits with redundant circuits are removed.

    :param circuit_list: a list of circuits
    :type circuit_list: list
    :return: a new list of circuits
    :rtype: list
    """
    new_circuit_list = []

    for new_circuit in circuit_list:
        if new_circuit_list:
            check_isomorphic = False

            for circuit in new_circuit_list:
                current_circuit = circuit.copy()
                current_circuit.unwrap_nodes()
                current_circuit.remove_identity()

                to_add_circuit = new_circuit.copy()
                to_add_circuit.unwrap_nodes()
                to_add_circuit.remove_identity()

                if circuit_is_isomorphic(current_circuit, to_add_circuit):
                    check_isomorphic = True
                    break
            if not check_isomorphic:
                new_circuit_list.append(new_circuit)
        else:
            new_circuit_list.append(new_circuit)

    return new_circuit_list


def check_redundant_circuit(circuit1, circuit2):
    circuit1_copy = circuit1.copy()
    circuit2_copy = circuit2.copy()

    circuit1_copy.unwrap_nodes()
    circuit2_copy.unwrap_nodes()

    circuit1_copy.remove_identity()
    circuit2_copy.remove_identity()

    return compare_circuits(circuit1_copy, circuit2_copy)


class CircuitStorage:
    """
    Class for storing circuits, check if circuit is duplicate before adding to circuit list
    """

    def __init__(self, check_function=None, disable_circuit_comparison=False):
        self.circuit_list = []
        self.disable_circuit_comparison = disable_circuit_comparison

        if check_function:
            self._check_func = check_function
        else:
            self._check_func = check_redundant_circuit

    def add_new_circuit(self, new_circuit):
        """
        Function to add new circuit to circuit list, check if new circuit is redundant.

        :param new_circuit: new circuit to be added
        :type new_circuit: CircuitDAG
        :return: Return True if added, False otherwise.
        :rtype: bool
        """
        if self.disable_circuit_comparison:
            self.circuit_list.append(new_circuit)
            return True

        if self.is_redundant(new_circuit):
            return False

        self.circuit_list.append(new_circuit)
        return True

    def is_redundant(self, new_circuit):
        """
        Function to check if circuit is redundant.

        :param new_circuit: circuit to be checked
        :type new_circuit: CircuitDAG
        :return: True if redundant, False otherwise.
        :rtype: bool
        """
        f = self._check_func

        if self.circuit_list and f:
            for circuit in self.circuit_list:
                if f(circuit, new_circuit):
                    return True
        return False
