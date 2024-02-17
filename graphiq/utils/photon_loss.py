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
Utility functions for analyzing the effect of photon loss on circuit performance.
"""

from graphiq.noise.noise_models import PhotonLoss


def photon_survival_rate(circuit):
    """
    Calculate the survival rate of photons

    The survival rate of a photon is the product of probabilities that this photon is not lost
    during a gate action

    :param circuit: circuit
    :type circuit: CircuitDAG
    :return: survival rate of photons
    :rtype: list
    """
    circuit = circuit.copy()
    circuit.unwrap_nodes()
    n_photons = circuit.n_photons
    survive = [1] * n_photons

    for reg in range(n_photons):
        out_node = f"p{reg}_out"
        node = f"p{reg}_in"

        while node != out_node:
            out_edge = [
                edge
                for edge in list(circuit.dag.out_edges(node, keys=True))
                if edge[2] == f"p{reg}"
            ]
            node = out_edge[0][1]
            op_noise = circuit.dag.nodes[node]["op"].noise
            if isinstance(op_noise, list):
                op_noise = op_noise[1]

            if isinstance(op_noise, PhotonLoss):
                survive[reg] = (1 - op_noise.noise_parameters["loss rate"]) * survive[
                    reg
                ]

    return survive
