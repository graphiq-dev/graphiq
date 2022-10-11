import copy
import src.noise.noise_models as nm

import src.ops as ops
import benchmarks.circuits as examples


def m_emissions_depolarizing_noise(circ, depolarizing_time, depolarizing_prob):
    emission_counts = {reg: 0 for reg in range(circ.n_emitters)}
    edges_to_add = []
    for edge in circ.edge_dict["e"]:
        reg, reg_type = circ.dag.edges[edge]["reg"], circ.dag.edges[edge]["reg_type"]
        start_node = circ.dag.nodes[edge[0]]
        if isinstance(start_node["op"], ops.CNOT):
            emission_counts[reg] += 1
            assert reg_type == "e"
            if emission_counts[reg] % depolarizing_time == 0:
                edges_to_add.append(edge)

    for edge in edges_to_add:
        reg, reg_type = circ.dag.edges[edge]["reg"], circ.dag.edges[edge]["reg_type"]
        noisy_id = ops.Identity(register=reg, reg_type=reg_type,
                                noise=nm.DepolarizingNoise(depolarizing_prob=depolarizing_prob))

        circ.insert_at(noisy_id, [edge])

    return circ


if __name__ == "__main__":
    circ, ideal_state = examples.ghz4_state_circuit()
    m_emissions_depolarizing_noise(circ, 1, 0.1)
    circ.draw_circuit()
