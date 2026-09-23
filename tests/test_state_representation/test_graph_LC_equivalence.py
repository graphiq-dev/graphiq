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
import time

import matplotlib.pyplot as plt
import pytest

from graphiq.benchmarks import graph_states as gs
from graphiq.backends.lc_equivalence_check import *
import graphiq.backends.lc_equivalence_check as lcc
import graphiq.backends.stabilizer.functions.linalg as slinalg
from graphiq.backends.graph.state import Graph
from graphiq.backends.lc_equivalence_check import _is_valid_clifford
from graphiq.backends.stabilizer.functions.local_cliff_equi_check import (
    lc_check,
    state_converter_circuit,
)
from graphiq.backends.stabilizer.functions.rep_conversion import (
    get_stabilizer_tableau_from_graph,
)
from graphiq.backends.stabilizer.functions.transformation import run_circuit
from graphiq.backends.stabilizer.tableau import StabilizerTableau
from graphiq.backends.state_rep_conversion import state_to_graph


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

    # nx.random_tree was removed in networkx 3.4 in favour of random_labeled_tree
    random_tree = getattr(nx, "random_tree", None) or nx.random_labeled_tree
    g = random_tree(12, seed=seed)
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


def test_state_to_graph_matrix_input_matches_graph_input():
    # lc_check must give the same answer whether the states are passed as
    # graphs or as their adjacency matrices; both go through state_to_graph.
    star4 = nx.star_graph(3)
    k4 = nx.complete_graph(4)

    result_from_graphs = lc_check(star4, k4)
    result_from_matrices = lc_check(nx.to_numpy_array(star4), nx.to_numpy_array(k4))

    assert result_from_graphs[0] is True
    assert result_from_matrices == result_from_graphs

    graph, tableau, gate_list = state_to_graph(nx.to_numpy_array(star4))
    assert isinstance(tableau, StabilizerTableau)
    assert gate_list == []


@pytest.mark.parametrize("mode", ["deterministic", "random"])
def test_is_lc_equivalent_star4_and_k4(mode):
    # a 4-node star and the complete graph on 4 nodes are LC equivalent
    star4 = nx.to_numpy_array(nx.star_graph(3)).astype(int)
    k4 = nx.to_numpy_array(nx.complete_graph(4)).astype(int)

    equivalent, solution = is_lc_equivalent(star4, k4, mode=mode, seed=0)

    assert equivalent
    n_nodes = star4.shape[0]
    assert _is_valid_clifford(solution.reshape(4 * n_nodes, 1))


def test_is_lc_equivalent_p4_and_its_complement_are_not_equivalent():
    # P4 and its complement are ruled out by the rank check in is_lc_equivalent
    # (the coefficient matrix has rank 4*n_nodes) before either search mode is
    # read, so this pair does not exercise the "deterministic"/"random" dispatch.
    p4 = nx.to_numpy_array(nx.path_graph(4)).astype(int)
    p4_complement = nx.to_numpy_array(nx.complement(nx.path_graph(4))).astype(int)

    equivalent, solution = is_lc_equivalent(
        p4, p4_complement, mode="deterministic", seed=0
    )

    assert not equivalent
    assert solution is None


@pytest.mark.parametrize("mode", ["deterministic", "random"])
def test_is_lc_equivalent_star5_and_star5_plus_edge_are_not_equivalent(mode):
    # star5 and star5 with one extra edge among its leaves pass the rank check
    # (rank 15 < 4*n_nodes = 20) with a solution-basis dimension of 5, so the
    # "deterministic"/"random" dispatch in is_lc_equivalent is actually reached,
    # and both searches correctly reject.
    star5 = nx.to_numpy_array(nx.star_graph(4)).astype(int)
    star5_plus_edge_graph = nx.star_graph(4)
    star5_plus_edge_graph.add_edge(1, 2)
    star5_plus_edge = nx.to_numpy_array(star5_plus_edge_graph).astype(int)

    equivalent, solution = is_lc_equivalent(star5, star5_plus_edge, mode=mode, seed=0)

    assert not equivalent
    assert solution is None


def test_find_lc_operations_recovers_first_graph_from_second():
    # These two 5-node graphs are LC equivalent via a multi-step local
    # complementation sequence (not a single node, unlike star4/k4), which is
    # needed to tell apart which of the two adjacency matrices the sequence
    # is built from.
    g1 = nx.Graph([(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (2, 4)])
    g2 = nx.Graph([(0, 1), (0, 2), (1, 3), (2, 4), (3, 4)])
    adj_matrix1 = nx.to_numpy_array(g1).astype(int)
    adj_matrix2 = nx.to_numpy_array(g2).astype(int)

    op_list = find_lc_operations(adj_matrix1, adj_matrix2, mode="deterministic", seed=0)

    # applying the returned local complementations to the second graph should
    # turn it into the first graph
    g = nx.from_numpy_array(adj_matrix2)
    for node in op_list:
        g = local_comp_graph(g, node)
    assert np.array_equal(nx.to_numpy_array(g).astype(int), adj_matrix1)


def _adjacency(graph):
    """
    Adjacency matrix of a networkx graph with a fixed, sorted node order, so both sides of a
    comparison index their nodes the same way.
    """
    return nx.to_numpy_array(graph, nodelist=sorted(graph.nodes)).astype(int)


def _solution_basis(z1, z2):
    """
    Mirrors the row-reduction/basis-finding steps of ``is_lc_equivalent`` on the whole (not
    per-component) system, so a test can enumerate every candidate solution independently of the
    algorithm under test.
    """
    coeff_matrix = lcc._coeff_maker(z1, z2)
    reduced_coeff_matrix, _, last_nonzero_row_index = slinalg.row_reduction(
        coeff_matrix, coeff_matrix * 0
    )
    n_nodes = np.shape(z1)[0]
    rank = last_nonzero_row_index + 1
    if rank >= 4 * n_nodes:
        return None
    reduced_coeff_matrix = np.array([row for row in reduced_coeff_matrix if row.any()])
    col_list = lcc._col_finder(reduced_coeff_matrix)
    return lcc._solution_basis_finder(reduced_coeff_matrix, col_list)


def _flat(sol):
    """
    Reshapes an (n, 2, 2) Clifford-operator array into the (4n, 1) vector ``_is_valid_clifford``
    expects.
    """
    return sol.reshape(-1, 1)


def test_disconnected_equivalent_pairs():
    """
    Minimal failing cases: the pair-sum search used to run on the whole graph instead of per
    connected component, so valid vectors that need contributions from more than one component's
    basis were missed.
    """
    # (a) two isolated vertices vs itself: two single-node components, each trivially equivalent
    # to itself via the identity.
    z_two_isolated = np.zeros((2, 2))
    equivalent, solution = is_lc_equivalent(z_two_isolated, z_two_isolated)
    assert equivalent
    identity = np.array([[1, 0], [0, 1]])
    assert np.array_equal(solution, np.stack([identity, identity]))

    # (b) two disjoint K3 vs two disjoint 3-stars: LC equivalent component-wise, but the pair-sum
    # search over the combined (6-node) system missed it.
    k3 = nx.complete_graph(3)
    star3 = nx.star_graph(2)
    two_k3 = nx.disjoint_union(k3, k3)
    two_star3 = nx.disjoint_union(star3, star3)
    z1, z2 = _adjacency(two_k3), _adjacency(two_star3)
    equivalent, solution = is_lc_equivalent(z1, z2)
    assert equivalent
    g_list = lc_graph_operations(z1, solution)
    g_new = two_k3
    for i in g_list:
        g_new = local_comp_graph(g_new, i)
    assert nx.is_isomorphic(g_new, two_star3), "found graphs are not LC equivalent"
    assert np.array_equal(
        nx.to_numpy_array(g_new), z2
    ), "found graphs are not LC equivalent"

    # (c) the same pair with an added isolated vertex on both sides (three components).
    two_k3_iso = nx.disjoint_union(two_k3, nx.empty_graph(1))
    two_star3_iso = nx.disjoint_union(two_star3, nx.empty_graph(1))
    z1_iso, z2_iso = _adjacency(two_k3_iso), _adjacency(two_star3_iso)
    equivalent, solution = is_lc_equivalent(z1_iso, z2_iso)
    assert equivalent
    g_list = lc_graph_operations(z1_iso, solution)
    g_new = two_k3_iso
    for i in g_list:
        g_new = local_comp_graph(g_new, i)
    assert nx.is_isomorphic(g_new, two_star3_iso), "found graphs are not LC equivalent"
    assert np.array_equal(
        nx.to_numpy_array(g_new), z2_iso
    ), "found graphs are not LC equivalent"

    # (d) the empty graph (0 nodes) vs itself. Only written after the fix: on unfixed code,
    # slinalg.row_reduction never terminates on a 0x0 matrix, and there is no timeout binary here
    # to cut a hung test.
    z_empty = np.zeros((0, 0), dtype=int)
    equivalent, solution = is_lc_equivalent(z_empty, z_empty)
    assert equivalent
    assert solution.shape == (0, 2, 2)


def test_component_partition_mismatch_is_false():
    """
    Two graphs whose adjacency matrices are the same size but partition into different connected
    components can never be LC equivalent (local complementation preserves connected components as
    vertex sets, Bouchet Section 6), and the check must reject them before running the linear-algebra
    search at all.
    """
    path4 = nx.path_graph(4)
    two_edges = nx.disjoint_union(nx.path_graph(2), nx.path_graph(2))
    equivalent, solution = is_lc_equivalent(_adjacency(path4), _adjacency(two_edges))
    assert not equivalent
    assert solution is None

    k3_plus_isolated = nx.disjoint_union(nx.complete_graph(3), nx.empty_graph(1))
    equivalent, solution = is_lc_equivalent(
        _adjacency(k3_plus_isolated), _adjacency(path4)
    )
    assert not equivalent
    assert solution is None


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_deterministic_seeded_sweep_keeps_disconnected_pairs(seed):
    """
    Sweeps G(n, 0.5) draws (edgeless graphs, the canonical failing shape, are not skipped) against
    an LC-equivalent partner reached by local complementation, and checks that the returned
    Clifford operator actually solves the coefficient system, without redrawing on failure (the
    redraw in the pre-existing ``_tester`` helper would hide exactly this bug).
    """
    rng = np.random.default_rng(seed)
    disconnected_draws = 0
    for n in range(4, 9):
        for _ in range(10):
            g = nx.gnp_random_graph(n, 0.5, seed=rng)
            if not nx.is_connected(g):
                disconnected_draws += 1
            lc_graphs, _ = gs.random_lc_equivalent(Graph(g.copy()), 1, 3, np_rng=rng)
            partner = lc_graphs[0].data
            z1 = _adjacency(g)
            z2 = _adjacency(partner)
            equivalent, sol = is_lc_equivalent(z1, z2, mode="deterministic")
            assert (
                equivalent
            ), f"seed={seed}, n={n}: partner via local complementation not found"
            assert lcc._is_valid_clifford(_flat(sol))
            assert np.all((lcc._coeff_maker(z1, z2) @ _flat(sol)) % 2 == 0)
    # Measured (this implementation): seed 0 -> 12, seed 1 -> 9, seed 2 -> 8, seed 3 -> 9 (of 50
    # draws each). G(n, 0.5) at n = 4 is disconnected about 40% of the time, so the sweep should
    # not go vacuous; a count this low would mean the draws stopped exercising the failing shape.
    assert (
        disconnected_draws >= 5
    ), f"seed {seed}: only {disconnected_draws} disconnected draws, sweep may be vacuous"


@pytest.mark.parametrize("seed", [0, 1])
def test_deterministic_matches_exhaustive_oracle(seed):
    """
    Compares ``is_lc_equivalent`` against a brute-force oracle (every candidate in the whole,
    not per-component, solution space) on a mix of LC-equivalent and independently-drawn pairs, so
    a regression that only breaks some shapes of input cannot hide behind a hand-picked example.
    """

    def oracle_equivalent(basis):
        if basis is None:
            return False
        k = len(basis)
        for bits in range(2**k):
            combo = np.zeros_like(basis[0])
            for j in range(k):
                if (bits >> j) & 1:
                    combo = (combo + basis[j]) % 2
            if lcc._is_valid_clifford(combo):
                return True
        return False

    rng = np.random.default_rng(seed)
    start = time.time()
    saw_nonequivalent_with_dim = False
    saw_disconnected_equivalent = False
    for n in range(4, 7):
        for i in range(20):
            g1 = nx.gnp_random_graph(n, 0.5, seed=rng)
            if i < 12:  # 60% LC partners
                lc_graphs, _ = gs.random_lc_equivalent(
                    Graph(g1.copy()), 1, 3, np_rng=rng
                )
                g2 = lc_graphs[0].data
            else:  # 40% independent draws
                g2 = nx.gnp_random_graph(n, 0.5, seed=rng)
            z1, z2 = _adjacency(g1), _adjacency(g2)
            basis = _solution_basis(z1, z2)
            dim = 0 if basis is None else len(basis)
            if dim > 12:  # keeps 2**dim well under 10 s
                continue
            equivalent, _ = is_lc_equivalent(z1, z2, mode="deterministic")
            assert equivalent == oracle_equivalent(basis), (seed, n, i, dim)
            if not equivalent and dim >= 1:
                saw_nonequivalent_with_dim = True
            if equivalent and not (nx.is_connected(g1) and nx.is_connected(g2)):
                saw_disconnected_equivalent = True
    wall_clock = time.time() - start
    # Measured (this implementation): seed 0 -> 0.08 s, seed 1 -> 0.07 s.
    assert wall_clock < 10
    assert saw_nonequivalent_with_dim
    assert saw_disconnected_equivalent


def test_candidates_include_single_basis_vectors(monkeypatch):
    """
    ``combinations(basis, 2)`` alone omits the k single basis vectors that Bouchet (4.4)'s "sum of
    at most two vectors of B" also includes; this is the only test that can catch a regression
    that removes the singles again (a differential sweep of 3000 pairs found no seed where the
    singles change the boolean answer).
    """
    star8 = nx.star_graph(7)  # 8 nodes; dim(solution basis) = 9, measured, connected.
    z = _adjacency(star8)
    basis = _solution_basis(z, z)
    k = len(basis)

    recorded = []

    def fake_is_valid_clifford(vector):
        recorded.append(vector.copy())
        return False  # never stop early, so every candidate is recorded

    monkeypatch.setattr(lcc, "_is_valid_clifford", fake_is_valid_clifford)
    equivalent, solution = is_lc_equivalent(z, z, mode="deterministic")
    assert not equivalent
    assert solution is None

    assert len(recorded) == k + k * (k - 1) // 2
    singles = recorded[:k]
    remaining_basis = list(basis)
    for single in singles:
        match = next(
            (i for i, b in enumerate(remaining_basis) if np.array_equal(b, single)),
            None,
        )
        assert (
            match is not None
        ), "recorded single basis-vector candidate not found in basis"
        remaining_basis.pop(match)
    assert remaining_basis == []


def test_symmetric_connected_graphs():
    """
    Regression guard, already passing before this change: a handful of large, connected,
    boundary-shaped comparisons (star, complete, complete-bipartite, path graphs) that should stay
    fast and correct.
    """
    start = time.time()
    star30 = nx.star_graph(29)
    k_15_15 = nx.complete_bipartite_graph(15, 15)
    equivalent, _ = is_lc_equivalent(_adjacency(star30), _adjacency(star30))
    assert equivalent
    equivalent, _ = is_lc_equivalent(_adjacency(k_15_15), _adjacency(k_15_15))
    assert equivalent

    star20 = nx.star_graph(19)
    k20 = nx.complete_graph(20)
    equivalent, _ = is_lc_equivalent(_adjacency(star20), _adjacency(k20))
    assert equivalent

    k_10_10 = nx.complete_bipartite_graph(10, 10)
    equivalent, _ = is_lc_equivalent(_adjacency(star20), _adjacency(k_10_10))
    assert not equivalent

    star40 = nx.star_graph(39)
    path40 = nx.path_graph(40)
    equivalent, _ = is_lc_equivalent(_adjacency(star40), _adjacency(path40))
    assert not equivalent

    wall_clock = time.time() - start
    # Measured (this implementation): 0.19 s.
    assert wall_clock < 20


def test_unknown_mode_raises_before_early_returns():
    """
    ``mode`` used to be validated only in the trailing ``else`` of the old single-function
    implementation, after the early returns for full rank and for a small solution-space
    dimension, so ``mode="bogus"`` silently returned a result on most inputs instead of raising.
    """
    # (a) two seeded random 6-node graphs whose system is full rank (no solution space at all).
    g1 = nx.gnp_random_graph(6, 0.5, seed=0)
    g2 = nx.gnp_random_graph(6, 0.5, seed=1000)
    z1, z2 = _adjacency(g1), _adjacency(g2)
    assert _solution_basis(z1, z2) is None
    with pytest.raises(ValueError):
        is_lc_equivalent(z1, z2, mode="bogus")

    # (b) path_8 vs itself: dimension 4, hits the exhaustive (< 5) branch's early return.
    path8 = nx.path_graph(8)
    z_path8 = _adjacency(path8)
    with pytest.raises(ValueError):
        is_lc_equivalent(z_path8, z_path8, mode="bogus")

    # (c) star_8 vs itself: dimension 9, green today (guard).
    star8 = nx.star_graph(7)
    z_star8 = _adjacency(star8)
    with pytest.raises(ValueError):
        is_lc_equivalent(z_star8, z_star8, mode="bogus")

    # (d) two isolated vertices vs itself: dimension 6 as a whole graph before the split, now two
    # one-node components of dimension 3 each; green today (guard).
    z_two_isolated = np.zeros((2, 2))
    with pytest.raises(ValueError):
        is_lc_equivalent(z_two_isolated, z_two_isolated, mode="bogus")


def test_state_converter_circuit_on_disconnected_pair():
    """
    Before this change, ``lc_check`` on a disconnected LC-equivalent pair returned ``(False, [])``:
    ``converter_gate_list`` asserted ``is_lc_equivalent``'s (buggy) False result and the bare
    ``except`` in ``lc_check`` swallowed the AssertionError.
    """
    two_k3 = nx.disjoint_union(nx.complete_graph(3), nx.complete_graph(3))
    two_star3 = nx.disjoint_union(nx.star_graph(2), nx.star_graph(2))
    result, gate_list = lc_check(two_k3, two_star3)
    assert result
    circuit = state_converter_circuit(two_k3, two_star3, validate=True)
    assert circuit is not None

    two_isolated = nx.empty_graph(2)
    circuit2 = state_converter_circuit(two_isolated, two_isolated, validate=True)
    assert circuit2 is not None
