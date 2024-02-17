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
Graph representation visualization helper
"""

import matplotlib.pyplot as plt
import networkx as nx


def draw_graph(state_graph, show=False, ax=None, with_labels=True):
    """
    Allows one to draw the underlying networkX graph with matplotlib library.

    :param state_graph: Graph state representation which we want to plot
    :type state_graph: Graph
    :param show: If True, draw the graph and show it. Otherwise, draw but do not show
    :type show: bool
    :param ax: axis on which to draw the graph
    :type ax: matplotlib.Axis
    :param with_labels: True if drawing the labels; False otherwise
    :type with_labels: bool
    :return: fig, axes on which the state is drawn
    :rtype: matplotlib.Figure, matplotlib.Axes
    """

    nx.draw(
        state_graph.data,
        with_labels=with_labels,
        font_weight="bold",
        ax=ax,
    )
    if show:
        plt.show()
    fig = plt.gcf()
    ax = plt.gca()
    return fig, ax
