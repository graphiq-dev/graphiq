import numpy as np
import networkx as nx
import warnings

from src.state import QuantumState
from src.backends.state_representation_conversion import _graph_finder as graph_finder
from src.backends.stabilizer.tableau import StabilizerTableau
from src.backends.stabilizer.tableau import CliffordTableau
from src.backends.stabilizer.functions.transformation import run_circuit
from src.backends.stabilizer.functions.stabilizer import canonical_form
from src.backends.stabilizer.functions.rep_conversion import (
    get_stabilizer_tableau_from_graph,
)
from src.backends.lc_equivalence_check import is_lc_equivalent, local_clifford_ops


def lc_check(state1, state2, validate=True):
    """
    Takes two quantum states (or stabilizer/clifford Tabeleaus, graphs, adjacency matrices) and checks the LC
    equivalence between them. If True, a sequence of gates to convert state1 to state 2 is also returned to.
    :param state1: the first state.
    :type state1: QuantumState or StabilizerTableau or CliffordTableau or nx.Graph or np.ndarray
    :param state2: the second state.
    :type state2: QuantumState or StabilizerTableau or CliffordTableau or nx.Graph or np.ndarray
    :param validate: if True, the provided list of gates is run on the state1 to confirm it ends up on state2 after the
    sequence.
    :return: tuple (result, gate_list)
    :rtype: tuple (bool, list)
    """
    # determines whether two states 1 and 2 are LC equivalent and if yes what is the gate sequence to turn 1 into 2.
    # if validate is True, function which check the gate list's
    graph1, tab1, gates1 = _to_graph(state1)
    graph2, tab2, gates2 = _to_graph(state2)
    try:
        gate_list = converter_circuit_list(graph1, graph2)
    except:
        return False, []
    # invert the sequence of gates that was used to convert state2 into graph2. Since we are converting state1 to state2
    # in the total gate list we must convert the graph2 to state 2 in the last step, hence the reversed gate list

    # replace P_dag operations with P in gates2, since they are inverse of each other. Other gates are self inverse
    # so no need to replace them. (and there is only X and P_dag gates in the gates1 and gates2 lists)
    inversed_gates2 = []
    for gate in gates2:
        if gate[0] == "P_dag":
            inversed_gates2.append(("P", gate[1]))
        else:
            inversed_gates2.append(gate)
    # reverse the order too
    inversed_gates2 = inversed_gates2[::-1]
    total_gate_list = gates1 + gate_list + inversed_gates2

    # validate
    if validate:
        new_tab = run_circuit(tab1, total_gate_list)
        # check if new_tab is the same as the stabilizer tableau for state2
        new_tab_can = canonical_form(new_tab)
        tab2_can = canonical_form(tab2)
        z_equal = np.array_equal(new_tab_can.z_matrix, tab2_can.z_matrix)
        x_equal = np.array_equal(new_tab_can.x_matrix, tab2_can.x_matrix)
        if not (z_equal and x_equal):
            warnings.warn(
                "the gate sequence is not converting the state1's stabilizer tableau into state2's"
            )

    return True, total_gate_list


def converter_circuit_list(g1, g2):
    """
    Given two graphs g1 and g2, this functions returns a sequence of gates needed to convert g1 into g2 if they are LC
     equivalent.
    :param g1: first graph
    :type g1: nx.Graph
    :param g2: second graph
    :type g2: nx.Graph
    :return: list of gates, which are (operation, qubit) tuple
    :rtype: list
    """
    adj_g1 = nx.to_numpy_array(g1)
    adj_g2 = nx.to_numpy_array(g2)
    lc, sol = is_lc_equivalent(adj_g1, adj_g2)
    assert lc, "the two graphs are not LC equivalent"
    lc_ops = local_clifford_ops(sol)
    circ_list = []
    for i, ops in enumerate(lc_ops):
        for op in ops.split()[::-1]:
            circ_list.append((op, i))
    return circ_list


def _to_graph(state):
    """
    A helper function to turn any valid representation into a graph. it also returns the StabilizerTableau corresponding
     to the initial state, and the gate sequence needed to convert the state into a graph-state.
    :param state: the state to be converted to graph
    :type state: QuantumState or StabilizerTableau or CliffordTableau or nx.Graph or np.ndarray
    :return: tuple (graph, stabilizer tableau, gate_list)
    :rtype: tup;e (nx.Graph, StabilizerTableau, list)
    """

    # returns graph, input state's tableau, gate_list
    if isinstance(state, nx.Graph):
        tab = get_stabilizer_tableau_from_graph(state)
        return state, tab, []
    elif isinstance(state, np.ndarray):
        try:
            graph = nx.from_numpy_array(state)
            tab = get_stabilizer_tableau_from_graph(graph)
            return graph, [], tab
        except:
            pass
    elif isinstance(state, QuantumState):
        try:
            stabilizer = state.stabilizer
        except:
            raise ValueError(
                "the QuantumState provided has no active stabilizer representation"
            )
        z_matrix = stabilizer.tableau.stabilizer_z
        x_matrix = stabilizer.tableau.stabilizer_x
        tab = StabilizerTableau([x_matrix, z_matrix])
    elif isinstance(state, CliffordTableau):
        z_matrix = state.stabilizer_z
        x_matrix = state.stabilizer_x
        tab = StabilizerTableau([x_matrix, z_matrix])
    elif isinstance(state, StabilizerTableau):
        z_matrix = state.z_matrix
        x_matrix = state.x_matrix
        tab = state
    else:
        raise ValueError(
            "input data should either be a adjacency matrix, graph, Clifford or Stabilizer tableau or a "
            "quantum state with stabilizer representation"
        )
    graph, (h_pos, xp_dag_pos) = graph_finder(x_matrix, z_matrix, get_ops_data=True)
    gate_list = [("H", pos) for pos in h_pos]
    for pos in xp_dag_pos:
        gate_list.append(("P_dag", pos))
        gate_list.append(("X", pos))
    return graph, tab, gate_list
