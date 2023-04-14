import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from src.backends.stabilizer.functions.height import height_max
from itertools import permutations
from networkx.algorithms import isomorphism
from src.backends.lc_equivalence_check import local_comp_graph


import warnings


def iso_finder(
    adj_matrix,
    n_iso,
    rel_inc_thresh=0.2,
    allow_exhaustive=True,
    sort_emit=False,
    label_map=False,
    thresh=None,
    seed=None,
):
    """
    The function permutes the labels of the vertices of a graph to get n_iso distinct isomorphic graphs. The original
    graph will also be returned as the first element of the list and counted toward the number of found cases.
    The maximum number of possible distinct cases may be less than n_iso. The graph G with the nodes relabeled using
    consecutive integers.

    :param adj_matrix: initial adjacency matrix or graph
    :type adj_matrix: numpy.ndarray
    :param n_iso: number of the isomorphic graphs to look for (including the original graph)
    :type n_iso: int
    :param rel_inc_thresh: a threshold value between 0 and 1. The closer to 0 the closer we get to an exhaustive search.
    :type rel_inc_thresh: float
    :param allow_exhaustive: whether exhaustive search over all possible permutations is allowed or not. The runtime may
    become too long if this parameter is True.
    :type allow_exhaustive: bool
    :param sort_emit: if True the outcome is sorted by the number of emitters needed to generate each graph
    :type sort_emit: bool
    :param label_map: if True, the function also returns a list of dictionaries, each representing the label mapping
     between the original input graph and the one in the returned list.
    :type label_map: bool
    :param thresh: a threshold value that determines how many random trials is performed in search for new permutations.
    The default is 5 times the number of needed permutations (=n_iso).
    :type thresh: int
    :param seed: the seed for random sampling of labels
    :type seed: int
    :return: An array of adjacency matrices. The length of the array may be less than n_iso since the maximum number of
    isomorphic graphs for the particular input may be reached. Or the function iteration is interrupted due to threshold
    values. If the label_map is True, function returns a tuple, the first item being the array of adjacency matrices and
     the second one a list of dictionaries that are label maps between the original and relabeled graphs.
    :rtype: numpy.ndarray or tuple (numpy.ndarray, list[dict])
    """
    n_node = adj_matrix.shape[0]
    n_max = np.math.factorial(n_node)
    n_label = n_iso
    labels_arr = _label_finder(n_label, n_node, seed=None, thresh=None)
    adj_arr = automorph_check(adj_matrix, labels_arr)
    if len(adj_arr) >= n_iso:
        adj_arr = adj_arr[:n_iso]
        return adj_arr
    else:
        rel_inc = 1  # relative increase ratio of the number of non-redundant cases as the number of labeling increase
        all_checked = False
        while len(adj_arr) < n_iso and rel_inc > rel_inc_thresh and not all_checked:
            success_ratio = len(adj_arr) / n_label if len(adj_arr) != 0 else 0.5
            add_n = int(n_label / success_ratio) + 1 - n_label
            n_label = int(n_label / success_ratio) + 1
            if n_label > n_max:
                n_label = n_max
                all_checked = True
                if (not allow_exhaustive) and success_ratio < 0.5:
                    warnings.warn(
                        f"Only {len(adj_arr)} isomorphic graphs were found due to high symmetry."
                        f"This may not be the maximum value. Allow for exhaustive search to check"
                    )
                    return adj_arr

            # update the seed used in _add_labels to get new values when we repeat it in the loop
            seed = seed + 1 if (seed is not None) else None
            labels_arr = _add_labels(
                labels_arr,
                add_n,
                exhaustive=allow_exhaustive * all_checked,
                seed=seed,
                thresh=thresh,
            )
            n1 = len(adj_arr)
            adj_arr = automorph_check(adj_matrix, labels_arr)
            n2 = len(adj_arr)
            if n2 >= n1:
                # update increase ratio and relative increase ratio
                inc_ratio = n2 / (n1 + 1)  # plus one is to avoid division by zero
                rel_inc = inc_ratio * success_ratio
        if all_checked:
            warnings.warn(f"Maximum of {n2} possible isomorphic graphs exist")
        elif rel_inc < rel_inc_thresh:
            warnings.warn(
                f"Only {n2} isomorphic graph were found. Consider decreasing rel_inc_thresh to possibly "
                f"get more."
            )
        else:
            pass
        if sort_emit:
            adj_arr = np.array([x[0] for x in emitter_sorted(adj_arr[:n_iso])])
        if label_map:
            mapping = []
            for new_adj in adj_arr:
                mapping.append(get_relabel_map(adj_matrix, new_adj))
            return adj_arr[:n_iso], mapping
        return adj_arr[:n_iso]


def emitter_sorted(adj_arr):
    """
    Takes an iterable of adjacency matrices (ideally output of the <iso_finder> function) as input and sort them by number of
    emitters needed to generate those graph states.

    :param adj_arr: An array of adjacency matrices which can be the output of the <iso_finder> function
    :type adj_arr: numpy.ndarray
    :return: a list of tuples (adj matrices, n_emitter) sorted by number of emitters
    :rtype: list
    """
    tuple_list = []
    for adj in adj_arr:
        g = nx.from_numpy_array(adj)
        n_emit = height_max(graph=g)
        tuple_list.append((adj, n_emit))
    sorted_list = sorted(tuple_list, key=lambda x: x[1])
    return sorted_list


def relabel(adj_matrix, new_labels):
    """
    This function relabels the vertices of a graph given the map to new labels. The initial graph's vertices is assumed
    to have been labeled according to its adjacency matrix that is with the nodes labeled using consecutive integers.

    :param adj_matrix: The adjacency matrix of the graph
    :type adj_matrix: numpy.ndarray
    :param new_labels: A permutation of the initial labels that is [0, 1, ..., n-1] where n is the number of vertices.
    :type new_labels: numpy.ndarray
    :return: Transformed adjacency matrix according to new labels
    :rtype: numpy.ndarray
    """
    p_matrix = _perm2matrix(new_labels)
    permuted_adj_matrix = p_matrix.T @ adj_matrix @ p_matrix
    return permuted_adj_matrix.astype(int)


def _perm2matrix(sequence):
    """
    Given a permutation of [0, 1, ..., n-1] where n is the number of vertices of the graph, this function returns the
    permutation matrix needed for the transformation of the adjacency matrix according to this label permutation.

    :param sequence: A list or 1-D array representing the permutation of graph vertices' labels.
    :type sequence: list or numpy.ndarray
    :return: The permutation matrix
    :rtype: numpy.ndarray
    """
    # we always consider the initial sequence to be ordered from 0 to n-1.
    n = len(sequence)
    permute_matrix = np.zeros([n, n])
    for i, label in enumerate(sequence):
        permute_matrix[i, label] = 1
    return permute_matrix


def _label_finder(
    n_label, n_node, new_label_set=None, exhaustive=False, seed=None, thresh=None
):
    """
    Finds n_label number of permutations for the sequence {0, 1, ..., n_node-1}. If exhaustive search is enabled the
    process happens through sampling from the total set of all possible permutations (size = n_node factorial).
    Otherwise, new permutations are generated randomly until the required number of distinct cases are found, or
    the threshold for number of random trials is reached.

    :param n_label: number of permutations needed
    :type n_label: int
    :param n_node: length of the initial sequence {0, 1, ..., n_node-1}
    :type n_node: int
    :param new_label_set: a given set of permutations that the function will add the new cases to.
    :type new_label_set: set
    :param exhaustive: If true the permutation will be sampled from the set of all possible permutations. There might be
     a significant increase in runtime for large sequences.
    :type exhaustive: bool
    :param seed: the seed for random sampling of labels
    :type seed: int
    :param thresh: threshold for the number of random trials to get permutations. This can be used to limit the runtime.
    :type thresh: int
    :return: an array of the found permutation
    :rtype: numpy.ndarray
    """
    rng = np.random.default_rng(seed)
    n_max = np.math.factorial(n_node)
    if thresh is None:
        thresh = 5 * n_label
    elif thresh < n_label:
        thresh = n_label + 1
    assert (
        n_label <= n_max
    ), f"The input number of permutations is more than the maximum possible"
    if n_node < 8 or exhaustive:
        perm = list(permutations([*range(n_node)]))
        initial_perm = np.array([[*range(n_node)]])
        labels_list = rng.choice(perm[1:], n_label - 1)
        labels_list = np.concatenate((initial_perm, labels_list), axis=0)
        return labels_list
    else:
        if new_label_set is None:
            new_label_set = set()
            new_label_set.add(tuple([*range(n_node)]))
        count = 0
        while len(new_label_set) < n_label and count < thresh:
            new_label_set.add(tuple(rng.permutation(n_node)))
            count += 1
        if count == thresh:
            warnings.warn(
                f"Only {len(new_label_set)} instead of {n_label} new permutations were found. If more"
                " is needed consider increasing the threshold."
            )
        new_label_list = list([list(x) for x in new_label_set])
        return np.array(new_label_list)


def _add_labels(labels_arr, add_n, exhaustive=False, seed=None, thresh=None):
    """
    Used to add new permutation to an initial array consisting of permutations of a sequence.

    :param labels_arr: an array of the permutation that new cases are added to
    :param add_n: number of new permutations needed
    :param exhaustive: If true the permutation will be sampled from the set of all possible permutations. There might be
     a significant increase in runtime for large sequences.
    :type exhaustive: bool
    :param seed: the seed for random sampling of labels
    :type seed: int
    :param thresh: threshold for the number of random trials to get permutations. This can be used to limit the runtime.
    :type thresh: int
    :return: an array of the found permutation
    :rtype: numpy.ndarray
    """
    n_node = labels_arr[0].shape[0]
    n_label = len(labels_arr)
    n_max = np.math.factorial(n_node)
    n_total = add_n + n_label if add_n + n_label < n_max else n_max
    label_set = set([tuple(labels) for labels in labels_arr])
    return _label_finder(
        n_total,
        n_node,
        new_label_set=label_set,
        exhaustive=exhaustive,
        thresh=thresh,
        seed=seed,
    )


def automorph_check(adj1, labels_arr):
    """
    Given an initial adjacency matrix (of a graph), and an array of new permutations of the nodes' labels, this function
    return an array of distinct adjacency matrices resulted from those permutations.
    Note that the order will not remain the same as the order of the input labels.

    :param adj1: initial adjacency matrix
    :type adj1: numpy.ndarray
    :param labels_arr:  an array of new labeling
    :type labels_arr: numpy.ndarray
    :return: An array of adjacency matrices
    :rtype: numpy.ndarray
    """
    # uses set to remove redundancies
    # the set includes the adjacency matrices in flatten form so that they can turn into tuples to be members of a set
    adj_set = {tuple(adj1.astype(int).flatten())}
    adj_list = [adj1]
    n_node = adj1.shape[0]
    for label in labels_arr:
        new_adj = relabel(adj1, label)
        adj_set.add(tuple(new_adj.flatten()))
    # remove the initial adjacency matrix from the set to just keep the new ones, since the original adj1 is already in
    # the final list of adjacencies as the first element
    adj_set.remove(tuple(adj1.astype(int).flatten()))
    for flat_adj in adj_set:
        remade_adj = np.array(flat_adj)
        adj_list.append(remade_adj.reshape(n_node, n_node))
    return np.array(adj_list)


def get_relabel_map(g1, g2):
    """
    Finds the map between nodes of g1 and g2 if they are isomorphic.

    :param g1: networkx graph or adjacency matrix
    :type g1: nx.Graph or np.ndarray
    :param g2: networkx graph or adjacency matrix
    :type g2: nx.Graph or np.ndarray
    :return: a dictionary mapping each node of g1 to a node of g2. The map may not be unique.
    :rtype: dict
    """
    # g1 and g2 can be adj matrices or nx graphs.
    if not isinstance(g1, nx.Graph):
        g1 = nx.from_numpy_array(g1)
        assert isinstance(g1, nx.Graph)
    if not isinstance(g2, nx.Graph):
        g2 = nx.from_numpy_array(g2)
        assert isinstance(g2, nx.Graph)
    GM = isomorphism.GraphMatcher(g1, g2)
    assert GM.is_isomorphic()
    return GM.mapping


def lc_orbit_finder(graph: nx.Graph, comp_depth=None, orbit_size_thresh=None):
    """
    Given a graph this functions tries all possible local-complementation sequences of length up to comp_depth to
    come up with new distinct graphs in the orbit of the input graph. The comp_depth determines the maximum depth of the
     orbit explored.

    :param graph: original graph
    :type graph: nx.Graph
    :param comp_depth: the maximum length of the sequence of local-complementations applied on the graph; if None,
                        continue till the required number of graphs are found, or no new graphs are found.
    :type comp_depth: int
    :param orbit_size_thresh: sets a limit on the maximum number of orbit graphs to look for
    :type orbit_size_thresh: int
    :return: list of distinct graphs in the orbit of original graph
    :rtype: list[nx.Graph]
    """
    orbit_list = [graph]
    if orbit_size_thresh == 1:
        return orbit_list
    new_graphs = 1
    i = 0
    if comp_depth is None:
        cond = lambda x: True
    else:
        cond = lambda x: bool(x < comp_depth)

    while cond(i):
        len_before = len(orbit_list)
        # iterate over the new graphs appended to the end of the orbit list
        for graph in orbit_list[-new_graphs:]:
            for node in graph.nodes:
                if graph.degree(node) > 1:
                    g_lc = local_comp_graph(graph, node)
                    if not check_isomorphism(g_lc, orbit_list):
                        orbit_list.append(g_lc)
                if (
                    orbit_size_thresh is not None
                    and len(orbit_list) >= orbit_size_thresh
                ):
                    return orbit_list[:orbit_size_thresh]
        # orbit_list = remove_iso(orbit_list)
        len_after = len(orbit_list)
        new_graphs = len_after - len_before
        # print("new graphs", new_graphs)
        if new_graphs == 0:
            break
        i += 1
    return orbit_list


def remove_iso(g_list):
    """
    Takes an input list of graphs and removes all the isomorphic cases, returning a list of distinct graphs

    :param g_list: list of graphs
    :type g_list: list[nx.Graph]
    :return: list of distinct graphs
    :rtype: list[nx.Graph]
    """
    non_iso = [*g_list]
    i, j = 0, 1
    while i < len(non_iso):
        g = non_iso[i]
        while i + j < len(non_iso):
            gg = non_iso[i + j]
            if nx.is_isomorphic(g, gg):
                del non_iso[i + j]
            else:
                j += 1
        i += 1
        j = 1
    return non_iso


def check_isomorphism(graph, g_list):
    """
    check if the provided graph is isomorphic to any graph in the g_list

    :param graph: graph to check
    :type graph: nx.Graph
    :param g_list: graph list to check against
    :type g_list: list
    :return: True of False, if an isomorphism case was detected.
    :rtype: bool
    """
    iso = False
    for g in g_list:
        if nx.is_isomorphic(graph, g):
            iso = True
            break
    return iso


def _compare_graphs_visual(g, new_g, new_labels):
    """
    Visual demonstration of initial graph, the relabeled one.
    """
    nx.draw_networkx(g, with_labels=True, pos=nx.kamada_kawai_layout(g))
    fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(15, 25))
    fig.tight_layout()
    n = len(new_labels)
    relabel_map_dict = dict(zip([*range(n)], new_labels))
    g_relabeled = nx.relabel_nodes(g, relabel_map_dict)
    assert list(new_labels) == g_relabeled.nodes()

    nx.draw_networkx(
        g_relabeled,
        node_size=1000,
        font_size=25,
        with_labels=True,
        ax=ax2,
        pos=nx.kamada_kawai_layout(g_relabeled),
    )
    nx.draw_networkx(
        new_g,
        node_size=1000,
        font_size=25,
        with_labels=True,
        ax=ax3,
        pos=nx.kamada_kawai_layout(g_relabeled),
    )
    nx.draw_networkx(
        g,
        node_size=1000,
        node_color="#bcbd22",
        font_size=25,
        with_labels=True,
        ax=ax1,
        pos=nx.kamada_kawai_layout(g),
    )
    plt.show()
    return
