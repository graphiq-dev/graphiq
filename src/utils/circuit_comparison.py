"""
Functions to compare quantum circuits

"""
# TODO: GED implementation requires further investigation.
import networkx as nx


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
                    and op1.c_registers == op2.c_registers
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
    dag1 = circuit1.dag
    dag2 = circuit2.dag

    def node_match(n1, n2):
        reg_match = (
            n1["op"].q_registers_type == n2["op"].q_registers_type
            and n1["op"].q_registers == n2["op"].q_registers
            and n1["op"].c_registers == n2["op"].c_registers
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
