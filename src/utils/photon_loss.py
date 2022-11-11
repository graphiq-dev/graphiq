from src.noise.noise_models import PhotonLoss


def photon_survival_rate(circuit):
    """
    Calculate the survival rate of photons

    :param circuit: circuit
    :type circuit: CircuitDAG
    :return: survival rate of photons
    :rtype: list
    """
    circuit = circuit.copy()
    circuit.unwrap_nodes()
    survive = [1]*len(circuit.register["p"])

    for reg in range(len(survive)):
        out_node = f"p{reg}_out"
        node = f"p{reg}_in"
        s = 1

        while node != out_node:
            out_edge = [
                edge
                for edge in list(circuit.dag.out_edges(node, keys=True))
                if edge[2] == f"p{reg}"
            ]
            node = out_edge[0][1]
            op_noise = circuit.dag.nodes[node]["op"].noise

            if isinstance(op_noise, PhotonLoss):
                s = (1 - op_noise.noise_parameters["loss rate"])*s

        survive[reg] = s

    return survive
