import numpy as np
from src.backends.stabilizer.tableau import CliffordTableau
from src.backends.stabilizer.functions.linalg import (
    multiply_columns,
    column_swap,
    add_columns,
    add_rows,
    row_reduction,
    row_sum,
)
from scipy.linalg import block_diag

"""Main gates """


def hadamard_gate(tableau, qubit_position):
    """
    hadamard gate applied on a single qubit given its position, in a stabilizer state.

    :param tableau: Tableau of the state before gate action
    :type tableau: CliffordTableau
    :param qubit_position: index of the qubit that the gate acts on
    :type qubit_position: int
    :return: the resulting state after gate action
    :rtype: CliffordTableau
    """
    n_qubits = tableau.n_qubits  # number of qubits
    assert qubit_position < n_qubits

    # update phase vector
    tableau.phase = tableau.phase ^ multiply_columns(
        tableau.table, tableau.table, qubit_position, n_qubits + qubit_position
    )
    # update the rest of the tableau
    tableau.table = column_swap(
        tableau.table, qubit_position, n_qubits + qubit_position
    )
    return tableau


def phase_gate(tableau, qubit_position):
    """
    Phase gate applied on the qubit given its position in a stabilizer state.

    :param tableau: Tableau of the state before gate action
    :type tableau: CliffordTableau
    :param qubit_position: index of the qubit that the gate acts on
    :type qubit_position: int
    :return: the resulting state after gate action
    :rtype: CliffordTableau
    """
    n_qubits = tableau.n_qubits  # number of qubits
    assert qubit_position < n_qubits

    # update phase vector
    tableau.phase = tableau.phase ^ multiply_columns(
        tableau.table, tableau.table, qubit_position, n_qubits + qubit_position
    )
    # update the rest of the tableau
    tableau.table = add_columns(
        tableau.table, qubit_position, n_qubits + qubit_position
    )
    return tableau


def cnot_gate(tableau, ctrl_qubit, target_qubit):
    """
    CNOT on control and target qubits given their position, in a stabilizer state.

    :param tableau: Tableau of the state before gate action
    :type tableau: CliffordTableau
    :param ctrl_qubit: index of the control qubit
    :type ctrl_qubit: int
    :param target_qubit: index of the target qubit that the gate acts on
    :type target_qubit: int
    :return: the resulting state after gate action
    :rtype: CliffordTableau
    """
    n_qubits = tableau.n_qubits  # number of qubits
    assert ctrl_qubit < n_qubits and target_qubit < n_qubits
    table = tableau.table

    # update phase vector
    x_ctrl_times_z_target = multiply_columns(
        table, table, ctrl_qubit, n_qubits + target_qubit
    )
    x_target = table[:, target_qubit]
    z_ctrl = table[:, n_qubits + ctrl_qubit]
    tableau.phase = tableau.phase ^ (x_ctrl_times_z_target * (x_target ^ z_ctrl ^ 1))

    # update the rest of the tableau
    tableau.table = add_columns(tableau.table, ctrl_qubit, target_qubit)
    tableau.table = add_columns(
        tableau.table, n_qubits + target_qubit, n_qubits + ctrl_qubit
    )

    return tableau


def z_measurement_gate(
    tableau, qubit_position, measurement_determinism="probabilistic"
):
    """
    Apply a Z-basis measurement on a given qubit in a stabilizer state.

    :param tableau: Tableau of the state before gate action
    :type tableau: CliffordTableau
    :param qubit_position: index of the qubit that the gate acts on
    :type qubit_position: int
    :param measurement_determinism: If the outcome is probabilistic from the simulation, we have the option to
        select a specific outcome.
    :type measurement_determinism: str or int
    :return: the resulting state after gate action, the measurement outcome, whether the measurement outcome is
        probabilistic (zero means deterministic)
    :rtype: CliffordTableau, int, int
    """
    n_qubits = tableau.n_qubits  # number of qubits
    assert qubit_position < n_qubits

    x_column = tableau.table[:, qubit_position]
    non_zero_x = np.nonzero(x_column)[0]
    x_p = 0
    # x_p is needed and important in other functions, so it will be returned as a function output
    for i in range(len(non_zero_x)):
        if non_zero_x[i] >= n_qubits:
            x_p = non_zero_x[i]
            non_zero_x = np.delete(non_zero_x, i)
            break
    if x_p != 0:
        # probabilistic outcome
        x_matrix = tableau.table_x
        z_matrix = tableau.table_z
        r_vector = tableau.phase
        # rowsum for all other x elements equal to 1 other than x_p
        for target_row in non_zero_x:
            x_matrix, z_matrix, r_vector = row_sum(
                x_matrix, z_matrix, r_vector, x_p, target_row
            )
        tableau.table_x = x_matrix
        tableau.table_z = z_matrix
        tableau.phase = r_vector
        table = tableau.table

        # set x_p - n row equal to x_p row
        table[x_p - n_qubits] = table[x_p]

        # set x_p row equal to 0 except for z element of measured qubit which is 1.
        table[x_p] = np.zeros(2 * n_qubits)
        table[x_p, qubit_position + n_qubits] = 1

        # set that row of the phase vector to random measurement outcome 0 or 1 with an equal probability.
        # We have the option to remove randomness and pick a specific outcome
        # measurement_determinism is only used when we know random measurement outcome is possible
        if measurement_determinism == "probabilistic":
            outcome = np.random.randint(0, 2)
        elif measurement_determinism == 1:
            outcome = 1
        else:
            outcome = 0

        tableau.phase[x_p] = outcome
        tableau.table = table
        return tableau, outcome, x_p
    else:
        # deterministic outcome
        # We ignore measurement_determinism here since the outcome is deterministic and fixed
        # add an extra 2n+1 th row to the tableau
        new_table = np.vstack([tableau.table, np.zeros(2 * n_qubits)]).astype(int)
        x_matrix = new_table[:, 0:n_qubits]
        z_matrix = new_table[:, n_qubits : 2 * n_qubits]
        r_vector = np.append(tableau.phase, 0).astype(int)

        # list of nonzero elements in the x destabilizers
        for non_zero in non_zero_x[non_zero_x < n_qubits]:
            x_matrix, z_matrix, r_vector = row_sum(
                x_matrix, z_matrix, r_vector, non_zero + n_qubits, 2 * n_qubits
            )
        # no need to update the tableau
        outcome = r_vector[2 * n_qubits]
        return tableau, outcome, x_p


"""SECONDARY GATES """


def phase_dagger_gate(tableau, qubit_position):
    """
    Phase dagger gate (inverse of phase) applied to a given qubit in a stabilizer state tableau.

    :param tableau: Tableau of the state before gate action
    :type tableau: CliffordTableau
    :param qubit_position: index of the qubit that the gate acts on
    :type qubit_position: int
    :return: the resulting state after gate action
    :rtype: CliffordTableau
    """

    for _ in range(3):
        tableau = phase_gate(tableau, qubit_position)

    return tableau


def z_gate(tableau, qubit_position):
    """
    Apply the Pauli Z to a given qubit in a stabilizer state tableau.

    :param tableau: Tableau of the state before gate action
    :type tableau: CliffordTableau
    :param qubit_position: index of the qubit that the gate acts on
    :type qubit_position: int
    :return: the resulting state after gate action
    :rtype: CliffordTableau
    """

    for _ in range(2):
        tableau = phase_gate(tableau, qubit_position)

    return tableau


def x_gate(tableau, qubit_position):
    """
    Apply the Pauli X (= HZH) to a given qubit in a stabilizer state tableau.

    :param tableau: Tableau of the state before gate action
    :type tableau: CliffordTableau
    :param qubit_position: index of the qubit that the gate acts on
    :type qubit_position: int
    :return: the resulting state after gate action
    :rtype: CliffordTableau
    """

    tableau = hadamard_gate(tableau, qubit_position)
    tableau = z_gate(tableau, qubit_position)
    tableau = hadamard_gate(tableau, qubit_position)

    return tableau


def y_gate(tableau, qubit_position):
    """
    Apply the Pauli Y (=PXZP) to a given qubit in a stabilizer state tableau.

    :param tableau: Tableau of the state before gate action
    :type tableau: CliffordTableau
    :param qubit_position: index of the qubit that the gate acts on
    :type qubit_position: int
    :return: the resulting state after gate action
    :rtype: CliffordTableau
    """

    tableau = phase_gate(tableau, qubit_position)
    tableau = z_gate(tableau, qubit_position)
    tableau = x_gate(tableau, qubit_position)
    tableau = phase_gate(tableau, qubit_position)

    return tableau


def control_z_gate(tableau, ctrl_qubit, target_qubit):
    """
    Controlled Z gate on control and target qubits in a stabilizer state tableau.

    :param tableau: Tableau of the state before gate action
    :type tableau: CliffordTableau
    :param ctrl_qubit: index of the control qubit
    :type ctrl_qubit: int
    :param target_qubit: index of the target qubit that the gate acts on
    :type target_qubit: int
    :return: the resulting state after gate action
    :rtype: CliffordTableau
    """
    tableau = hadamard_gate(tableau, target_qubit)
    tableau = cnot_gate(tableau, ctrl_qubit, target_qubit)
    tableau = hadamard_gate(tableau, target_qubit)
    return tableau


def control_y_gate(tableau, ctrl_qubit, target_qubit):
    """
    Controlled Y gate on control and target qubits in a stabilizer state tableau.

    :param tableau: Tableau of the state before gate action
    :type tableau: CliffordTableau
    :param ctrl_qubit: index of the control qubit
    :type ctrl_qubit: int
    :param target_qubit: index of the target qubit that the gate acts on
    :type target_qubit: int
    :return: the resulting state after gate action
    :rtype: CliffordTableau
    """
    tableau = phase_gate(tableau, target_qubit)
    tableau = z_gate(tableau, target_qubit)
    tableau = cnot_gate(tableau, ctrl_qubit, target_qubit)
    tableau = phase_gate(tableau, target_qubit)
    return tableau


def reset_z(
    tableau, qubit_position, intended_state, measurement_determinism="probabilistic"
):
    """
    Resets a qubit to a Z basis state. Note that it only works after a measurement gate on the same qubit or if
    the qubit is isolated. Otherwise, the action of this gate would be measuring in Z basis and resetting the qubit.

    :param tableau: Tableau of the state before gate action
    :type tableau: CliffordTableau
    :param qubit_position: index of the qubit to be reset
    :type qubit_position: int
    :param intended_state: either 0 for :math:`|0 \\rangle` state or 1 for :math:`|1 \\rangle`  state
    :type intended_state: int
    :param measurement_determinism:
    :type measurement_determinism: int or str
    :return: updated tableau
    :rtype:CliffordTableau
    """
    # reset qubit to computational basis states

    n_qubits = tableau.n_qubits  # number of qubits
    assert qubit_position < n_qubits
    assert intended_state == 0 or intended_state == 1
    tableau, outcome, probabilistic = z_measurement_gate(
        tableau, qubit_position, measurement_determinism
    )
    if probabilistic:
        tableau.phase[probabilistic] = intended_state
        return tableau

    else:
        if outcome == intended_state:
            return tableau
        else:
            return x_gate(tableau, qubit_position)
            # the following code does differently and leave here for debugging purpose.
            # non_zero = [i for i in range(n_qubits)if tableau.destabilizer_x[i, qubit_position] != 0]
            # if len(non_zero) <= 1:
            #     tableau.phase[non_zero[0]] = 1 ^ tableau.phase[non_zero[0]]
            #     return tableau
            # else:
            #     removed_qubit_table = np.delete(
            #         tableau.table, [qubit_position, n_qubits + qubit_position], axis=1
            #     )
            #     for i in non_zero:
            #         if np.array_equal(
            #             removed_qubit_table[i], np.zeros(2 * n_qubits - 2)
            #         ):
            #             tableau.phase[i] = 1 ^ tableau.phase[i]
            #             return tableau
            #     tableau.phase[non_zero[-1]] = 1 ^ tableau.phase[non_zero[-1]]
            #     return tableau


def reset_x(
    tableau, qubit_position, intended_state, measurement_determinism="probabilistic"
):
    """
    Reset the qubit in one of X-basis states

    :param tableau:
    :type tableau: CliffordTableau
    :param qubit_position:
    :type qubit_position: int
    :param intended_state:
    :type intended_state: int
    :param measurement_determinism:
    :type measurement_determinism: str or int
    :return:
    :rtype: CliffordTableau
    """
    new_tableau = reset_z(
        tableau,
        qubit_position,
        intended_state,
        measurement_determinism=measurement_determinism,
    )
    new_tableau = hadamard_gate(new_tableau, qubit_position)
    return new_tableau


def reset_y(
    tableau, qubit_position, intended_state, measurement_determinism="probabilistic"
):
    """
    Reset the qubit to one of the eigenstates of the Y basis.

    :param tableau:
    :type tableau: CliffordTableau
    :param qubit_position:
    :type qubit_position: int
    :param intended_state:
    :type intended_state: int
    :param measurement_determinism:
    :type measurement_determinism: str or int
    :return:
    :rtype: CliffordTableau
    """
    new_tableau = reset_z(
        tableau,
        qubit_position,
        intended_state,
        measurement_determinism=measurement_determinism,
    )
    new_tableau = hadamard_gate(new_tableau, qubit_position)
    new_tableau = phase_gate(new_tableau, qubit_position)
    return new_tableau


def add_qubit(tableau):
    """
    Add one isolated qubit in :math:`|0 \\rangle` state to the current state at the end.

    :param tableau: the input state tableau
    :type tableau: CliffordTableau
    :return: the updated stabilizer state
    :rtype: CliffordTableau
    """
    return insert_qubit(tableau, tableau.n_qubits)


def insert_qubit(tableau, new_position):
    """
    Insert a qubit in :math:`| 0 \\rangle` state to a given position.

    :param tableau: the state represented by a CliffordTableau before insertion
    :type tableau: CliffordTableau
    :param new_position: the future position of the inserted qubit
    :type new_position: int
    :return: updated state
    :rtype: CliffordTableau
    """
    n_qubits = tableau.n_qubits
    assert new_position < n_qubits
    new_column = np.zeros(n_qubits)
    new_row = np.zeros(n_qubits + 1)
    # x destabilizer part

    tmp_dex = np.insert(tableau.destabilizer_x, new_position, new_column, axis=1)
    tmp_dex = np.insert(tmp_dex, new_position, new_row, axis=0)

    # z destabilizer part
    tmp_dez = np.insert(tableau.destabilizer_z, new_position, new_column, axis=1)
    tmp_dez = np.insert(tmp_dez, new_position, new_row, axis=0)

    # x stabilizer part
    tmp_sx = np.insert(tableau.stabilizer_x, new_position, new_column, axis=1)
    tmp_sx = np.insert(tmp_sx, new_position, new_row, axis=0)

    # z stabilizer part
    tmp_sz = np.insert(tableau.stabilizer_z, new_position, new_column, axis=1)
    tmp_sz = np.insert(tmp_sz, new_position, new_row, axis=0)

    # phase vector part
    new_phase = np.insert(tableau.phase, [new_position, n_qubits + 1 + new_position], 0)

    new_table = np.block([[tmp_dex, tmp_dez], [tmp_sx, tmp_sz]])
    tableau.expand(new_table, new_phase)

    # set the new qubit to ket 0 state
    tableau.destabilizer_x[new_position, new_position] = 1
    tableau.stabilizer_z[new_position, new_position] = 1

    return tableau


def remove_qubit(tableau, qubit_position, measurement_determinism="probabilistic"):
    """
    The action of the function is measure and remove. If isolated, state should not change. Entangled qubits cannot be
    removed without affecting other parts of the state.
    Only works correctly for isolated qubits! e.g. after measurement.
    TODO: Check if a qubit is isolated in general. Only isolated qubits in the Z basis states can be confirmed for now.

    :param tableau:
    :type tableau: CliffordTableau
    :param qubit_position:
    :type qubit_position: int
    :param measurement_determinism:
    :type measurement_determinism: str or int
    :return:
    :rtype: CliffordTableau
    """
    n_qubits = tableau.n_qubits  # number of qubits
    assert qubit_position < n_qubits
    tableau, outcome, probabilistic = z_measurement_gate(
        tableau, qubit_position, measurement_determinism
    )
    new_table = np.delete(
        tableau.table, [qubit_position, qubit_position + n_qubits], axis=1
    )

    if probabilistic:
        new_table = np.delete(
            new_table, [probabilistic, probabilistic - n_qubits], axis=0
        )
        new_phase = np.delete(tableau.phase, [probabilistic, probabilistic - n_qubits])
    else:
        non_zero = [
            i for i in range(n_qubits) if tableau.destabilizer_x[i, qubit_position] != 0
        ]
        if len(non_zero) <= 1:
            new_table = np.delete(
                new_table, [non_zero[0], non_zero[0] + n_qubits], axis=0
            )
            new_phase = np.delete(tableau.phase, [non_zero[0], non_zero[0] + n_qubits])
        else:
            new_phase = tableau.phase
            for i in non_zero:
                if np.array_equal(new_table[i], np.zeros(2 * n_qubits - 2)):
                    new_table = np.delete(new_table, [i, i + n_qubits], axis=0)
                    new_phase = np.delete(new_phase, [i, i + n_qubits])
            new_table = np.delete(
                new_table, [non_zero[-1], non_zero[-1] + n_qubits], axis=0
            )
            new_phase = np.delete(new_phase, [non_zero[-1], non_zero[-1] + n_qubits])

    tableau.shrink(new_table, new_phase)
    return tableau


def measure_x(tableau, qubit_position, measurement_determinism="probabilistic"):
    """
    Returns the outcome 0 or 1 if one measures the given qubit in the X basis.
    NOTE: cannot update the stabilizer state after measurement. Stabilizer formalism can only handle Z-measurements.

    :param tableau:
    :type tableau: CliffordTableau
    :param qubit_position: index of the qubit that the gate acts on
    :type qubit_position: int
    :return: the classical outcome of measuring given qubit in the X basis.
    :rtype: int
    """
    stabilizer_state_new = hadamard_gate(tableau, qubit_position)
    _, outcome, _ = z_measurement_gate(
        stabilizer_state_new, qubit_position, measurement_determinism
    )
    return outcome


def measure_y(tableau, qubit_position, measurement_determinism="probabilistic"):
    """
    Returns the outcome 0 or 1 if one measures the given qubit in the Y basis.
    NOTE: cannot update the stabilizer state after measurement. Stabilizer formalism can only handle Z-measurements.

    :param tableau:
    :type tableau: CliffordTableau
    :param qubit_position: index of the qubit that the gate acts on
    :type qubit_position: int
    :return: the classical outcome of measuring given qubit in the Y basis.
    :rtype: int
    """
    new_tableau = tableau
    # apply P dagger gate
    new_tableau = phase_dagger_gate(new_tableau, qubit_position)
    # apply H
    new_tableau = hadamard_gate(new_tableau, qubit_position)

    _, outcome, _ = z_measurement_gate(
        new_tableau, qubit_position, measurement_determinism
    )
    return outcome


def measure_z(tableau, qubit_position, measurement_determinism="probabilistic"):
    """
    Returns the outcome 0 or 1 if one measures the given qubit in the Z basis.
    NOTE: Does not update the stabilizer state after measurement.

    :param tableau: the initial state tableau
    :type tableau: CliffordTableau
    :param qubit_position: index of the qubit that the gate acts on
    :type qubit_position: int
    :param measurement_determinism:
    :type measurement_determinism: str or int
    :return: the classical outcome of measuring given qubit in the Z basis.
    :rtype: int
    """
    _, outcome, _ = z_measurement_gate(tableau, qubit_position, measurement_determinism)
    return outcome


def swap_gate(tableau, qubit1, qubit2):
    """
    Swap gate between two qubits.

    :param tableau: a stabilizer state tableau consists of stabilizer and destabilizer
    :type tableau: CliffordTableau
    :param qubit1: One of the qubits as input to the swap gate
    :type qubit1: int
    :param qubit2: The other qubit position
    :type qubit2: int
    :return: Updated state
    :rtype: CliffordTableau
    """
    n_qubits = tableau.n_qubits  # number of qubits
    assert qubit1 < n_qubits and qubit2 < n_qubits
    tableau.table = column_swap(tableau.table, qubit1, qubit2)
    tableau.table = column_swap(tableau.table, qubit1 + n_qubits, qubit2 + n_qubits)
    tableau.phase[[qubit1, qubit2]] = tableau.phase[[qubit2, qubit1]]
    tableau.phase[[qubit1 + n_qubits, qubit2 + n_qubits]] = tableau.phase[
        [qubit2 + n_qubits, qubit1 + n_qubits]
    ]
    return tableau


def create_n_ket0_state(n_qubits):
    """
    Create the product state :math:`|0\\rangle^{\\otimes n}`

    :param n_qubits: number of qubits
    :type n_qubits: int
    :return: the state :math:`|0\\rangle^{\\otimes n}`
    :rtype: CliffordTableau
    """
    return CliffordTableau(n_qubits)


def create_n_plus_state(n_qubits):
    """
    Create the product state :math:`|+\\rangle^{\\otimes n}`

    :param n_qubits: number of qubits
    :type n_qubits: int
    :return: the state :math:`|+\\rangle^{\\otimes n}`
    :rtype: CliffordTableau
    """
    tableau = create_n_ket0_state(n_qubits)

    tableau.table[:, [*range(2 * n_qubits)]] = tableau.table[
        :, [*range(n_qubits, 2 * n_qubits)] + [*range(0, n_qubits)]
    ]

    return tableau


def tensor(list_of_tables):
    """
    Return the stabilizer state (Clifford) tableau that is tensor product of states given by list_of_tables

    :param list_of_tables: A list of the states' tableau we want to tensor in the same order (left to right)
    :type list_of_tables: list[CliffordTableau]
    :return: The resulting tableau
    :rtype: CliffordTableau
    """
    tableau = list_of_tables[0]
    list_of_tables = list_of_tables[1:]
    for tab in list_of_tables:
        tableau.n_qubits = tableau.n_qubits + tab.n_qubits
        tableau.destabilizer.x_matrix = block_diag(
            tableau.destabilizer.x_matrix, tab.destabilizer.x_matrix
        )
        tableau.destabilizer.z_matrix = block_diag(
            tableau.destabilizer.z_matrix, tab.destabilizer.z_matrix
        )
        tableau.stabilizer.x_matrix = block_diag(
            tableau.stabilizer.x_matrix, tab.stabilizer.x_matrix
        )
        tableau.stabilizer.z_matrix = block_diag(
            tableau.stabilizer.z_matrix, tab.stabilizer.z_matrix
        )
        phase_list1 = np.split(tableau.phase, 2)
        phase_list2 = np.split(tab.phase, 2)
        phase_vector = np.hstack(
            (phase_list1[0], phase_list2[0], phase_list1[1], phase_list2[1])
        ).astype(int)

        tableau.phase = phase_vector
    return tableau


def partial_trace(tableau, keep, dims, measurement_determinism="probabilistic"):
    """
    Return the tableau corresponding to taking the partial trace of the state

    :param tableau: the state represented by CliffordTableau
    :type tableau: CliffordTableau
    :param keep: the qubit positions to be kept
    :type keep: list[int]
    :param dims: currently not used for this function
    :type dims: list[int]
    :param measurement_determinism:
    :type measurement_determinism:
    :return: the tableau corresponding to taking the partial trace of the state
    :rtype: CliffordTableau
    """

    n_qubits = tableau.n_qubits
    total = set(range(n_qubits))
    keep = set(keep)
    removal = sorted(total - keep, reverse=True)
    for qubit_position in removal:
        tableau = remove_qubit(tableau, qubit_position, measurement_determinism)

    return tableau


def full_rank_x(tableau):
    """
    Based on lemma 6 in arXiv:quant-ph/0406196

    :param tableau:
    :type tableau:
    :return:
    :rtype:
    """

    n_qubits = tableau.n_qubits
    x_matrix = tableau.stabilizer_x
    z_matrix = tableau.stabilizer_z
    x_matrix, z_matrix, index = row_reduction(x_matrix, z_matrix)
    rank = index + 1
    z1_matrix = z_matrix[rank:n_qubits, 0:rank]
    z2_matrix = z_matrix[rank:n_qubits, rank:n_qubits]
    z2_matrix, z1_matrix, index1 = row_reduction(z2_matrix, z1_matrix)
    assert index1 == n_qubits - rank - 1
    for j in range(n_qubits - rank):
        for i in range(j):
            if z2_matrix[i, j] == 1:
                z2_matrix = add_rows(z2_matrix, j, i)
                z1_matrix = add_rows(z1_matrix, j, i)

    assert np.array_equal(z2_matrix, np.eye(n_qubits - rank))
    z_matrix = np.hstack(z1_matrix, z2_matrix).astype(int)
    tableau.x_stabilizer = x_matrix
    tableau.z_stabilizer = z_matrix
    # hadamard on some qubits to make the x-stabilizer table full rank
    for qubit in range(rank, n_qubits):
        tableau = hadamard_gate(tableau, qubit)

    return tableau


def run_circuit(tableau, circuit_list):
    """
    Return the stabilizer state tableau after the execution of the circuit.

    :param tableau: initial state tableau
    :type tableau: CliffordTableau
    :param circuit_list: a list of gates in the circuit
    :type circuit_list: list[tuple]
    :return: the stabilizer state tableau after the execution of the circuit.
    :rtype: CliffordTableau
    """
    for ops in circuit_list:
        if ops[0] == "H":
            tableau = hadamard_gate(tableau, ops[1])
        if ops[0] == "P":
            tableau = phase_gate(tableau, ops[1])
        if ops[0] == "CNOT":
            tableau = cnot_gate(tableau, ops[1], ops[2])
        if ops[0] == "CZ":
            tableau = control_z_gate(tableau, ops[1], ops[2])
    return tableau
