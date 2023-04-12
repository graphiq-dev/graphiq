import numpy as np
import networkx as nx
import copy

from src.state import QuantumState
from src.circuit import CircuitDAG
from src.metrics import Infidelity
import src.ops as ops

from src.backends.stabilizer.compiler import StabilizerCompiler
from src.backends.state_representation_conversion import _graph_finder as graph_finder
from src.backends.stabilizer.tableau import StabilizerTableau, CliffordTableau
from src.backends.stabilizer.functions.transformation import run_circuit
from src.backends.stabilizer.functions.stabilizer import canonical_form
from src.backends.stabilizer.functions.rep_conversion import (
    get_stabilizer_tableau_from_graph,
    stabilizer_from_clifford,
    clifford_from_stabilizer,
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
    :type validate: bool
    :return: tuple (result, gate_list)
    :rtype: tuple (bool, list)
    """
    # determines whether two states 1 and 2 are LC equivalent and if yes what is the gate sequence to turn 1 into 2.
    # if validate is True, function which check the gate list's
    graph1, tab1, gates1 = _to_graph(state1)
    graph2, tab2, gates2 = _to_graph(state2)

    try:
        gate_list = converter_gate_list(graph1, graph2)
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
        elif gate[0] == "P":
            inversed_gates2.append(("P_dag", gate[1]))
        else:
            inversed_gates2.append(gate)
    # reverse the order too
    inversed_gates2 = inversed_gates2[::-1]

    total_gate_list = gates1 + gate_list + inversed_gates2

    # validate
    if validate:
        # check if new_tab is the same as the stabilizer tableau for state2
        tab2_can = canonical_form(tab2.copy())
        final_tab = run_circuit(tab1.copy(), total_gate_list)
        final_tab_can = canonical_form(final_tab)
        z_equal = np.array_equal(final_tab_can.z_matrix, tab2_can.z_matrix)
        x_equal = np.array_equal(final_tab_can.x_matrix, tab2_can.x_matrix)
        phase_equal = np.array_equal(final_tab_can.phase, tab2_can.phase)
        if not (z_equal and x_equal and phase_equal):
            raise Warning(
                "the gate sequence is not converting the state1's stabilizer tableau into state2's"
            )

    return True, total_gate_list


def state_converter_circuit(state1, state2, validate=False):
    """
    This function returns a piece of circuit that converts state1 to state2; if they are local clifford equivalent.
    The circuit corresponds to the gate_list which is returned by the function 'lc_check'.
    :param state1: the initial state
    :type state1: QuantumState or StabilizerTableau or CliffordTableau or nx.Graph or np.ndarray
    :param state2: the target state
    :type state2: QuantumState or StabilizerTableau or CliffordTableau or nx.Graph or np.ndarray
    :param validate: if True, validates the circuit to see if final output of the circuit is equal to the desired state.
    :type validate: bool
    :return: the circuit to convert graph1 to graph2
    :rtype: CircuitDAG
    """
    # s1 = copy.deepcopy(state1)
    # s2 = copy.deepcopy(state2)

    graph1, tab1, gates1 = _to_graph(state1)
    graph2, tab2, gates2 = _to_graph(state2)

    target1_tableau = clifford_from_stabilizer(tab1)
    target2_tableau = clifford_from_stabilizer(tab2)
    n_photon = target1_tableau.n_qubits

    lc, gate_list = lc_check(state1, state2)
    assert lc, "the two graphs are not LC equivalent!"

    circuit = CircuitDAG(n_photon=n_photon)
    for gate in gate_list:
        op = str_to_op(gate)
        circuit.add(op)
    if validate:
        target1 = QuantumState(n_photon, target1_tableau, representation="stabilizer")
        target2 = QuantumState(n_photon, target2_tableau, representation="stabilizer")
        metric2 = Infidelity(target2)
        compiler = StabilizerCompiler()
        final_state = compiler.compile(circuit, initial_state=target1)
        final_score = metric2.evaluate(final_state, circuit)
        assert np.isclose(
            final_score, 0
        ), "the final compiled state is not equal to the desired target state"

    return circuit


def converter_gate_list(g1, g2):
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
    gate_list = []
    for i, ops in enumerate(lc_ops):
        for op in ops.split()[::-1]:
            gate_list.append((op, i))
    tab1 = get_stabilizer_tableau_from_graph(g1)
    tab2 = get_stabilizer_tableau_from_graph(g2)
    phase_correction = _phase_correction(tab1, tab2, gate_list)
    gate_list += phase_correction
    return gate_list


def str_to_op(gate_tuples):
    """
    Converts a gate list, made up of gate tuples ("gate name", qubit index) used in the stabilizer backend, into a list
    of operations ready to be added to a circuit. Qubits are photons by default, not emitters.
    :param gate_tuples: a gate tuple or list of tuples
    :type gate_tuples: tuple or list
    :return: an operation or list of operations
    :rtype: ops.OneQubitOperationBase or list
    """
    # converts the (single qubit gate name, qubit index) into an OneQubitOperationBase object.
    # input gate is a tuple (name, qubit index) or a list of them
    name_list = ["I", "H", "X", "P", "P_dag", "Z"]
    ops_list = [
        ops.Identity,
        ops.Hadamard,
        ops.SigmaX,
        ops.Phase,
        ops.PhaseDagger,
        ops.SigmaZ,
    ]
    if isinstance(gate_tuples, tuple) and len(gate_tuples) == 2:
        op_index = name_list.index(gate_tuples[0])
        operation = ops_list[op_index](register=gate_tuples[1], reg_type="p")
        return operation
    elif isinstance(gate_tuples, list):
        operations_list = []
        for gate in gate_tuples:
            op_index = name_list.index(gate[0])
            operations_list.append(ops_list[op_index](register=gate[1], reg_type="p"))
        return operations_list
    else:
        raise ValueError(
            "input should be gate tuples ('gate name', qubit index) or a list of them"
        )


def _to_graph(state):
    """
    A helper function to turn any valid representation into a graph. it also returns the StabilizerTableau corresponding
     to the initial state, and the gate sequence needed to convert the state into a graph-state.
    :param state: the state to be converted to graph
    :type state: QuantumState or StabilizerTableau or CliffordTableau or nx.Graph or np.ndarray
    :return: tuple (graph, input state's stabilizer tableau, gate_list)
    :rtype: tuple (nx.Graph, StabilizerTableau, list)
    """
    state = copy.deepcopy(state)
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
            clifford = state.stabilizer.tableau
        except:
            raise ValueError(
                "the QuantumState provided has no active stabilizer representation"
            )
        z_matrix = clifford.stabilizer_z
        x_matrix = clifford.stabilizer_x
        tab = stabilizer_from_clifford(clifford)
    elif isinstance(state, CliffordTableau):
        z_matrix = state.stabilizer_z
        x_matrix = state.stabilizer_x
        tab = stabilizer_from_clifford(state)
    elif isinstance(state, StabilizerTableau):
        z_matrix = state.z_matrix
        x_matrix = state.x_matrix
        tab = state
    else:
        raise ValueError(
            "input data should either be a adjacency matrix, graph, Clifford or Stabilizer tableau or a "
            "quantum state with stabilizer representation"
        )
    graph, (h_pos, p_dag_pos) = graph_finder(x_matrix, z_matrix, get_ops_data=True)
    gate_list = [("H", pos) for pos in h_pos] + [("P_dag", pos) for pos in p_dag_pos]

    # phase correction; adding Z gates at the end to make the phase of the transformed state equal to an ideal graph
    g_tab = get_stabilizer_tableau_from_graph(graph)
    phase_correction = _phase_correction(tab, g_tab, gate_list)

    gate_list += phase_correction
    return graph, tab, gate_list


def _phase_correction(stabilizer_tab1, stabilizer_tab2, gate_list):
    """
    If gate list transforms stabilizer generators of state 1 to state 2, then this function finds the list of gates
     needed to be added to the gate list to also have the phase of the 2 states exactly the same.
    The second state's stabilizer generators must represent a graph state.
     Returns a set of Z gates applied on appropriate qubits in the format of list of tuples [("Z", qubit_index)]
    :param stabilizer_tab1: stabilizer tableau for the initial state
    :type stabilizer_tab1: StabilizerTableau
    :param stabilizer_tab2: stabilizer tableau for the final state
    :type stabilizer_tab2: StabilizerTableau
    :param gate_list: a gate list, made up of gate tuples ("gate name", qubit index) that transforms initial state's
     stabilizer
    :type gate_list: list
    :return: a list of tuples [("Z", qubit_index)] to correct phase
    """

    tab1 = canonical_form(stabilizer_tab1)
    tab2 = canonical_form(stabilizer_tab2)
    new_tab = canonical_form(run_circuit(tab1.copy(), gate_list))
    phase_diff = (tab2.phase - new_tab.phase) % 2
    x_mat = np.copy(new_tab.x_matrix)
    x_inv = ((np.linalg.det(x_mat) * np.linalg.inv(x_mat)) % 2).astype(int)
    z_ops = (x_inv @ phase_diff) % 2
    phase_correction = [("Z", index) for index, z in enumerate(z_ops) if z]
    return phase_correction
