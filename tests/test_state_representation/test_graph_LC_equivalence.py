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
import matplotlib.pyplot as plt
import pytest

from graphiq.benchmarks import graph_states as gs
from graphiq.backends.lc_equivalence_check import *
from graphiq.backends.stabilizer.functions.local_cliff_equi_check import (
    lc_check,
    state_converter_circuit,
)
from graphiq.backends.stabilizer.functions.rep_conversion import (
    get_stabilizer_tableau_from_graph,
)
from graphiq.backends.stabilizer.functions.transformation import run_circuit


def _tester(n):
    """
    A test function for internal purpose. Searches over all random graphs of size "n" to finally find two that are
        LC equivalent. Should not be used for n > 7 since it may never find a solution in reasonable time.

    :param n: the size of the graphs (number of the nodes/ qubits)
    :type n: int
    :return: solution, G1, G2. Two graphs that are LC equivalent and the Clifford operator needed for the transformation
    :rtype: numpy.ndarray, networkx.Graph, networkx.Graph
    """
    solution = None
    while not isinstance(solution, np.ndarray):
        g1, g2 = nx.fast_gnp_random_graph(n, 0.65), nx.fast_gnp_random_graph(n, 0.65)

        z_1 = nx.to_numpy_array(g1).astype(int)
        z_2 = nx.to_numpy_array(g2).astype(int)

        success, solution = is_lc_equivalent(z_1, z_2, mode="deterministic")

    return solution, (g1, g2)


@pytest.mark.parametrize("n_nodes", [2, 3, 4])
def test_equivalence(n_nodes):
    solution, (graph1, graph2) = _tester(n_nodes)
    z_1 = nx.to_numpy_array(graph1).astype(int)

    g_list = lc_graph_operations(z_1, solution)
    print("LC operations needed on nodes:", g_list)
    g_new = graph1
    for i in g_list:
        g_new = local_comp_graph(g_new, i)

    z_new = nx.to_numpy_array(g_new)
    z_2 = nx.to_numpy_array(graph2).astype(int)
    assert nx.is_isomorphic(g_new, graph2), "found graphs are not LC equivalent"
    assert np.array_equal(z_new, z_2), "found graphs are not LC equivalent"


@pytest.mark.parametrize("n_nodes", [2, 3, 4, 5, 6])
def test_star_graph(n_nodes):
    g1 = nx.complete_graph(n_nodes)
    g2 = nx.star_graph(n_nodes - 1)
    z_1 = nx.to_numpy_array(g1).astype(int)
    z_2 = nx.to_numpy_array(g2).astype(int)
    success, solution = is_lc_equivalent(z_1, z_2, mode="deterministic")
    if isinstance(solution, np.ndarray):
        print(local_clifford_ops(solution), "\n")
    assert isinstance(solution, np.ndarray) or solution is None
    is_equivalent, _ = iso_equal_check(g1, g2)
    assert is_equivalent


def _lc_equiv_test(graph, seed, n_graphs=5, max_transform_path=17):
    """
    Make local complementation operations, and verify that is_lc_equivalent
    reports the fact that the original and final graphs are the same
    """
    new_graphs, new_graph_seqs = gs.random_lc_equivalent(
        graph, n_graphs, max_transform_path, np_rng=np.random.default_rng(seed=seed)
    )
    for new_graph, new_graph_seq in zip(new_graphs, new_graph_seqs):
        equivalent, solution = graph.lc_equivalent(new_graph, mode="deterministic")
        assert equivalent, (
            "Graph built from complementations was assessed as not LC equivalent. "
            f"Complementation sequence was on nodes: {new_graph_seq}"
        )


@pytest.mark.parametrize("seed", [0, 335, 930])
def test_equivalence_random_lc_linear(seed):
    graph = gs.linear_cluster_state(10)
    _lc_equiv_test(graph, seed)


@pytest.mark.parametrize("seed", [0, 335, 930])
def test_equivalence_random_lc_star(seed):
    graph = gs.star_graph_state(25)
    _lc_equiv_test(graph, seed)


@pytest.mark.parametrize("seed", [0, 335, 930])
def test_equivalence_random_lc_lattice_1(seed):
    graph = gs.lattice_cluster_state((7, 3, 5))
    _lc_equiv_test(graph, seed)


@pytest.mark.parametrize("seed", [0])
def test_equivalence_random_lc_lattice_debug_1(seed):
    graph = gs.lattice_cluster_state((6, 3))
    fig, ax = plt.subplots(figsize=(10, 10))
    _lc_equiv_test(graph, seed, n_graphs=1, max_transform_path=4)


@pytest.mark.parametrize("seed", [2])
def test_equivalence_random_lc_lattice_debug_2(seed):
    graph = gs.lattice_cluster_state((2, 3))
    fig, ax = plt.subplots(figsize=(10, 10))
    _lc_equiv_test(graph, seed, n_graphs=1, max_transform_path=4)


@pytest.mark.parametrize("seed", [0, 2, 4, 6, 8, 10, 11])
def test_random_state_converter(seed):
    """
    Apply a series of LC operations on a random graph states to transfer it to another stabilizer state. Then use
    relevant functions to check whether the two states are LC equivalent and find out the gates and circuit needed to
    convert one to the other.
    """

    g = nx.random_tree(12, seed)
    # also use the parameter seed to determine which node to apply local complementation on
    gg = local_comp_graph(g, seed)
    tab1 = get_stabilizer_tableau_from_graph(g)
    tab2 = get_stabilizer_tableau_from_graph(gg)
    random_tab1 = run_circuit(
        tab1.copy(),
        [("X", 2), ("Y", 3), ("H", 1), ("P_dag", 3), ("X", 4), ("X", 0), ("P", 1)],
    )
    random_tab2 = run_circuit(
        tab2.copy(),
        [
            ("P_dag", 0),
            ("Y", 2),
            ("P_dag", 2),
            ("Y", 1),
            ("H", 4),
            ("P_dag", 3),
            ("Z", 4),
            ("X", 0),
            ("P", 1),
            ("H", 3),
            ("X", 0),
        ],
    )
    assert lc_check(random_tab1, random_tab2)[0]
    c1 = state_converter_circuit(g, gg, validate=True)
    c2 = state_converter_circuit(random_tab1, random_tab2, validate=True)
    # check if output is not None
    assert c1 and c2
