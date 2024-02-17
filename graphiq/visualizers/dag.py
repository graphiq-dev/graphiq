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
Functions for visualizing the circuit DAGs
"""

import networkx as nx


def dag_topology_pos(dag, method="topology"):
    """
    Returns the node positions for plotting a DAG structure

    The methods are:
        topology: uses the requirement of being a DAG to order nodes from left to right in the given topological order
        spring: uses the default networkx spring layout option

    :param dag: the DAG structure which is going to be plotted
    :param method: 'topology' or 'spring' (see above)
    :type method: str
    :return: a position dictionary (key: name of a node, value: node position as a tuple (x, y) )
    :rtype: dict
    """
    if method == "topology":
        """Display in topological order, with simple offsetting for legibility"""
        pos_dict = {}
        for i, node_list in enumerate(nx.topological_generations(dag)):
            y_offset = len(node_list) / 2
            x_offset = 0.1
            for j, name in enumerate(node_list):
                pos_dict[name] = (i - j * x_offset, j - y_offset)

    elif method == "spring":
        pos_dict = nx.spring_layout(dag, seed=0)  # Seed layout for reproducibility
    else:
        raise ValueError("Only available plotting options are 'spring' and 'topology'")

    return pos_dict
