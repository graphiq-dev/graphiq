"""
Functions to compare quantum circuits
"""
import networkx as nx
import src.ops as ops

def compare(circuit_1, circuit_2, method="direct"):
    """
    Comparing two circuits by using GED or direct loop method

    :param circuit_1: circuit that to be compared
    :type circuit_1: CircuitDAG
    :param circuit_2: circuit that to be compared
    :type circuit_2: CircuitDAG
    :param method: Determine which comparison function to use
    :type method: str
    :return: whether two circuits are the same
    :rtype: bool
    """
    if method == "direct":
        return direct(circuit_1, circuit_2)
    elif method == "GED_full":
        return ged(circuit_1, circuit_2, full=True)
    elif method == "GED_approximate":
        return ged(circuit_1, circuit_2, full=False)
    elif method == "GED_adaptive":
        return ged_adaptive(circuit_1, circuit_2)
    else:
        raise ValueError(f"Method {method} is not supported.")

def direct(circuit_1, circuit_2):
    """
    Directly compare two circuits by iterating from input nodes to output nodes

    :param circuit_1: circuit that to be compared
    :type circuit_1: CircuitDAG
    :param circuit_2: circuit that to be compared
    :type circuit_2: CircuitDAG
    :return: whether two circuits are the same
    :rtype: bool
    """
    circuit_1 = circuit_1.copy()
    circuit_1.unwrap_nodes()
    circuit_2 = circuit_2.copy()
    circuit_2.unwrap_nodes()
    n_reg_match = circuit_1.register == circuit_2.register
    n_nodes_match = circuit_1.dag.number_of_nodes() == circuit_2.dag.number_of_nodes()

    if n_reg_match and n_nodes_match:
        for i in circuit_1.node_dict["Input"]:
            node_1 = i
            node_2 = i
            reg = list(circuit_1.dag.out_edges(i, keys=True))[0][2]
            while node_1 not in circuit_1.node_dict["Output"]:
                out_edge = [
                    edge
                    for edge in list(circuit_1.dag.out_edges(node_1, keys=True))
                    if edge[2] == reg
                ]
                node_1 = out_edge[0][1]

                out_edge_compare = [
                    edge
                    for edge in list(
                        circuit_2.dag.out_edges(node_2, keys=True)
                    )
                    if edge[2] == reg
                ]
                node_2 = out_edge_compare[0][1]

                op_1 = circuit_1.dag.nodes[node_1]["op"]
                op_2 = circuit_2.dag.nodes[node_2]["op"]
                control_match = (
                        op_1.q_registers_type == op_2.q_registers_type
                        and op_1.q_registers == op_2.q_registers
                        and op_1.c_registers == op_2.c_registers
                )
                if isinstance(op_1, type(op_2)) and control_match:
                    pass
                else:
                    return False
        return True
    else:
        return False


def ged_adaptive(circuit_1, circuit_2, threshold=30):
    """
    switch between exact and approximate GED calculation adaptively

    :param circuit_1: circuit that to be compared
    :type circuit_1: CircuitDAG
    :param circuit_2: circuit that to be compared
    :type circuit_2: CircuitDAG
    :param threshold: threshold
    :type threshold: int
    :return: exact/approximated GED between circuits(cost needed to transform self.dag to circuit_compare.dag)
    :rtype: bool
    """

    full = (
            max(circuit_1.dag.number_of_nodes(), circuit_2.dag.number_of_nodes())
            < threshold
    )
    sim = ged(circuit_1, circuit_2, full=full)
    return sim


def ged(circuit_1, circuit_2, full=True):
    """
    Calculate Graph Edit Distance (GED) between two circuits.
    Further reading on GED:
    https://networkx.org/documentation/stable/reference/algorithms/similarity.html

    :param circuit_1: circuit that to be compared
    :type circuit_1: CircuitDAG
    :param circuit_2: circuit that to be compared
    :type circuit_2: CircuitDAG
    :param full: Determine which GED function to use
    :type full: bool
    :return: whether two circuits are the same
    :rtype: bool
    """

    dag_1 = circuit_1.modify_dag_for_ged()
    dag_2 = circuit_2.modify_dag_for_ged()

    def node_subst_cost(n1, n2):
        reg_match = (
                n1["op"].q_registers_type == n2["op"].q_registers_type
                and n1["op"].c_registers == n2["op"].c_registers
        )
        ops_match = isinstance(n1["op"], type(n2["op"]))

        if reg_match and ops_match:
            if isinstance(n1["op"], ops.Input) and n1["op"].reg_type == "p":
                p_reg_match = n1["op"].register == n2["op"].register
                return int(not p_reg_match)
            else:
                return 0
        else:
            return 1

    def edge_subst_cost(e1, e2):
        if e1["control"] == e2["control"]:
            return 0
        else:
            return 1

    if full:
        sim = nx.algorithms.similarity.graph_edit_distance(
            dag_1,
            dag_2,
            node_subst_cost=node_subst_cost,
            edge_subst_cost=edge_subst_cost,
            upper_bound=30,
            timeout=10.0,
        )
    else:
        sim = nx.algorithms.similarity.optimize_graph_edit_distance(
            dag_1,
            dag_2,
            node_subst_cost=node_subst_cost,
            edge_subst_cost=edge_subst_cost,
            upper_bound=30,
        )
        sim = next(sim)

    return sim == 0


