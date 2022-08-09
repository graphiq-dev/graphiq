import numpy as np
from src.backends.stabilizer.tableau import CliffordTableau
from src.backends.stabilizer.functions.matrix_functions import (
    multiply_columns,
    column_swap,
    add_columns,
    row_sum,
    add_rows,
    row_reduction,
)
from scipy.linalg import block_diag


"""Main gates """


def hadamard_gate(tableau, qubit_position):
    """
    hadamard gate applied on a single qubit given its position, in a stabilizer state.

    :param tableau:
    :type tableau: CliffordTableau
    :param qubit_position: index of the qubit that the gate acts on
    :type qubit_position: int
    :return: the resulting state after gate action
    :rtype: CliffordTableau
    """
    n_qubits = tableau.n_qubits  # number of qubits
    assert qubit_position < n_qubits

    # updating phase vector
    tableau.phase = tableau.phase ^ multiply_columns(
        tableau.table, tableau.table, qubit_position, n_qubits + qubit_position
    )
    # updating the rest of the tableau
    tableau.table = column_swap(
        tableau.table, qubit_position, n_qubits + qubit_position
    )

    return tableau


def phase_gate(tableau, qubit_position):
    """
    Phase gate applied on the qubit given its position in a stabilizer state.

    :param tableau:
    :type tableau: CliffordTableau
    :param qubit_position: index of the qubit that the gate acts on
    :type qubit_position: int
    :return: the resulting state after gate action
    :rtype: CliffordTableau
    """
    n_qubits = tableau.n_qubits  # number of qubits
    assert qubit_position < n_qubits

    # updating phase vector
    tableau.phase = tableau.phase ^ multiply_columns(
        tableau.table, tableau.table, qubit_position, n_qubits + qubit_position
    )
    # updating the rest of the tableau
    tableau.table = add_columns(
        tableau.table, qubit_position, n_qubits + qubit_position
    )

    return tableau


def cnot_gate(tableau, ctrl_qubit, target_qubit):
    """
    CNOT on control and target qubits given their position, in a stabilizer state.

    :param tableau:
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

    # updating phase vector
    x_ctrl_times_z_target = multiply_columns(
        table, table, ctrl_qubit, n_qubits + target_qubit
    )
    x_target = table[:, target_qubit]
    z_ctrl = table[:, n_qubits + ctrl_qubit]
    tableau.phase = tableau.phase ^ (x_ctrl_times_z_target * (x_target ^ z_ctrl ^ 1))

    # updating the rest of the tableau
    table[:, target_qubit] = table[:, target_qubit] ^ table[:, ctrl_qubit]
    table[:, n_qubits + ctrl_qubit] = (
        table[:, n_qubits + ctrl_qubit] ^ table[:, n_qubits + target_qubit]
    )
    tableau.table = table
    return tableau


def z_measurement_gate(
    tableau, qubit_position, measurement_determinism="probabilistic"
):
    """
    Measurement applied on a single qubit given its position in a stabilizer state.

    :param tableau:
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
        # set r_vector of that row to random measurement outcome 0 or 1. (equal probability)
        # We have the option to remove randomness and pick a specific outcome
        # measurement_determinism is only used when we know random measurement outcome is possible
        if measurement_determinism == "probabilistic":
            outcome = np.random.randint(0, 1)
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
        new_table = np.vstack([tableau.table, np.zeros(2 * n_qubits)])
        x_matrix = new_table[:, 0:n_qubits]
        z_matrix = new_table[:, n_qubits : 2 * n_qubits]
        r_vector = np.append(tableau.phase, 0)

        # list of nonzero elements in the x destabilizers
        non_zero_x = [i for i in non_zero_x if i < n_qubits]
        for non_zero in non_zero_x:
            x_matrix, z_matrix, r_vector = row_sum(
                x_matrix, z_matrix, r_vector, non_zero + n_qubits, 2 * n_qubits
            )
        # no need to update the tableau
        outcome = r_vector[2 * n_qubits]
        return tableau, outcome, x_p


"""SECONDARY GATES """


def phase_dagger_gate(tableau, qubit_position):
    """
    Phase dagger gate (inverse of phase) applied on a single qubit given its position, in a stabilizer state.

    :param tableau:
    :type tableau: CliffordTableau
    :param qubit_position: index of the qubit that the gate acts on
    :type qubit_position: int
    :return: the resulting state after gate action
    :rtype: CliffordTableau
    """
    n_qubits = tableau.n_qubits  # number of qubits
    assert qubit_position < n_qubits
    for _ in range(3):
        tableau = phase_gate(tableau, qubit_position)

    return tableau


def z_gate(tableau, qubit_position):
    """
    Pauli Z applied on a single qubit given its position, in a stabilizer state tableau.

    :param tableau:
    :type tableau: CliffordTableau
    :param qubit_position: index of the qubit that the gate acts on
    :type qubit_position: int
    :return: the resulting state after gate action
    :rtype: CliffordTableau
    """
    n_qubits = tableau.n_qubits  # number of qubits
    assert qubit_position < n_qubits
    for _ in range(2):
        tableau = phase_gate(tableau, qubit_position)

    return tableau


def x_gate(tableau, qubit_position):
    """
    Pauli X (= HZH) applied on a single qubit given its position, in a stabilizer state tableau.

    :param tableau:
    :type tableau: CliffordTableau
    :param qubit_position: index of the qubit that the gate acts on
    :type qubit_position: int
    :return: the resulting state after gate action
    :rtype: CliffordTableau
    """
    n_qubits = tableau.n_qubits  # number of qubits
    assert qubit_position < n_qubits
    tableau = hadamard_gate(tableau, qubit_position)
    tableau = z_gate(tableau, qubit_position)
    tableau = hadamard_gate(tableau, qubit_position)

    return tableau


def y_gate(tableau, qubit_position):
    """
    Pauli Y (=PXZP) applied on a single qubit given its position, in a stabilizer state tableau.

    :param tableau:
    :type tableau: CliffordTableau
    :param qubit_position: index of the qubit that the gate acts on
    :type qubit_position: int
    :return: the resulting state after gate action
    :rtype: CliffordTableau
    """
    n_qubits = tableau.n_qubits  # number of qubits
    assert qubit_position < n_qubits
    tableau = phase_gate(tableau, qubit_position)
    tableau = z_gate(tableau, qubit_position)
    tableau = x_gate(tableau, qubit_position)
    tableau = phase_gate(tableau, qubit_position)

    return tableau


def control_x_gate(tableau, ctrl_qubit, target_qubit):
    """
    Controlled X gate on control and target qubits given their position, in a stabilizer state.

    :param tableau:
    :type tableau: CliffordTableau
    :param ctrl_qubit: index of the control qubit
    :type ctrl_qubit: int
    :param target_qubit: index of the target qubit that the gate acts on
    :type target_qubit: int
    :return: the resulting state after gate action
    :rtype: CliffordTableau
    """
    return cnot_gate(tableau, ctrl_qubit, target_qubit)


def control_z_gate(tableau, ctrl_qubit, target_qubit):
    """
    Controlled Z gate on control and target qubits given their position, in a stabilizer state.

    :param tableau:
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
    Controlled Y gate on control and target qubits given their position, in a stabilizer state.

    :param tableau:
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


def projector_z0(tableau, qubit_position, measurement_determinism="probabilistic"):
    """
    This function is probably not needed.

    :param tableau:
    :type tableau:
    :param qubit_position:
    :type qubit_position:
    :param measurement_determinism:
    :type measurement_determinism:
    :return:
    :rtype:
    """
    success = True
    n_qubits = tableau.n_qubits  # number of qubits
    assert qubit_position < n_qubits
    tableau, outcome, probabilistic = z_measurement_gate(
        tableau, qubit_position, measurement_determinism
    )
    if probabilistic:
        if outcome == 0:
            pass
        else:
            tableau.phase[probabilistic] = 0
    else:
        if outcome == 0:
            pass
        else:
            tableau = 0
            success = False
            # TODO: see how impossible projection is handled in the density matrix formalism
    return tableau, success


def projector_z1(tableau, qubit_position, measurement_determinism="probabilistic"):
    """
     This function is probably not needed.

    :param tableau:
    :type tableau:
    :param qubit_position:
    :type qubit_position:
    :param measurement_determinism:
    :type measurement_determinism:
    :return:
    :rtype:
    """
    success = True
    n_qubits = tableau.n_qubits  # number of qubits
    assert qubit_position < n_qubits
    tableau, outcome, probabilistic = z_measurement_gate(
        tableau, qubit_position, measurement_determinism
    )
    if probabilistic:
        if outcome == 1:
            pass
        else:
            tableau.phase[probabilistic] = 1
    else:
        if outcome == 1:
            pass
        else:
            tableau = 0
            success = False
            # TODO: see how impossible projection is handled in the density matrix formalism
    return tableau, success


def reset_z(
    tableau, qubit_position, intended_state, measurement_determinism="probabilistic"
):
    """
    Resets a qubit to a Z basis state. Note that it only works after a measurement gate on the same qubit or for
    isolated qubits. Otherwise, the action of this gate would be like measure in Z basis and reset.

    :param tableau: Tableau of the state before gate action
    :type tableau: CliffordTableau
    :param qubit_position: index of the qubit to be reset
    :type qubit_position: int
    :param intended_state: either 0 for Z_0 state or 1 for Z_1 state
    :type intended_state: int
    :param measurement_determinism:
    :type measurement_determinism: int or str
    :return: updated tableau
    :rtype:CliffordTableau
    """
    # reset qubit to computational basis states
    # .
    n_qubits = tableau.n_qubits  # number of qubits
    assert qubit_position < n_qubits
    assert intended_state == 0 or intended_state == 1
    tableau, outcome, probabilistic = z_measurement_gate(
        tableau, qubit_position, measurement_determinism
    )
    if probabilistic:
        if outcome == intended_state:
            return tableau
        else:
            tableau.phase[probabilistic] = intended_state
            return tableau
    else:
        if outcome == intended_state:
            return tableau
        else:
            non_zero = [
                i
                for i in range(n_qubits)
                if tableau.destabilizer_x[i, qubit_position] != 0
            ]
            if len(non_zero) <= 1:
                tableau.phase[non_zero[0]] = 1 ^ tableau.phase[non_zero[0]]
                return tableau
            else:
                removed_qubit_table = np.delete(
                    tableau.table, [qubit_position, n_qubits + qubit_position], axis=1
                )
                for i in non_zero:
                    if np.array_equal(
                        removed_qubit_table[i], np.zeros(2 * n_qubits - 2)
                    ):
                        tableau.phase[i] = 1 ^ tableau.phase[i]
                        return tableau
                tableau.phase[non_zero[-1]] = 1 ^ tableau.phase[non_zero[-1]]
                return tableau


def reset_x(
    tableau, qubit_position, intended_state, measurement_determinism="probabilistic"
):
    """


    :param tableau:
    :type tableau:
    :param qubit_position:
    :type qubit_position:
    :param intended_state:
    :type intended_state:
    :param measurement_determinism:
    :type measurement_determinism:
    :return:
    :rtype:
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


    :param tableau:
    :type tableau:
    :param qubit_position:
    :type qubit_position:
    :param intended_state:
    :type intended_state:
    :param measurement_determinism:
    :type measurement_determinism:
    :return:
    :rtype:
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
    add one isolated qubit in :math:`|0 \\rangle` state to the current state at the end.
    # TODO: rewrite this by calling insert_qubit

    :return: the updated stabilizer state
    :rtype: CliffordTableau
    """
    n_qubits = tableau.n_qubits  # number of qubits
    n_new = 1 + n_qubits
    if n_qubits == 0:
        return create_n_ket0_state(1)

    new_tableau = create_n_ket0_state(n_new)
    # x destabilizer part
    new_tableau.destabilizer_x[0:n_qubits, 0:n_qubits] = tableau.destabilizer_x
    # z destabilizer part
    new_tableau.destabilizer_z[0:n_qubits, 0:n_qubits] = tableau.destabilizer_z
    # r destabilizer part
    new_tableau.phase[0:n_qubits] = tableau.phase[0, n_qubits]
    # x stabilizer part
    new_tableau.stabilizer_x[0:n_qubits, 0:n_qubits] = tableau.stabilizer_x
    # z stabilizer part
    new_tableau.stabilizer_z[0:n_qubits, 0:n_qubits] = tableau.stabilizer_z
    # r stabilizer part
    new_tableau.phase[n_new : 2 * n_new - 1] = tableau.phase[n_qubits : 2 * n_qubits]

    # TODO: decide whether we want to update the old tableau
    tableau.expand(new_tableau.table, new_tableau.phase)

    return new_tableau


def insert_qubit(tableau, new_qubit_position):
    """
    Insert a qubit in :math:`| 0 \\rangle` state to a given position.

    :param tableau: the state represented by a CliffordTableau before insertion
    :type tableau: CliffordTableau
    :param new_qubit_position: the future position of the inserted qubit
    :type new_qubit_position: int
    :return: updated state
    :rtype: CliffordTableau
    """
    n_qubits = tableau.n_qubits
    assert new_qubit_position < n_qubits
    new_column = np.zeros(n_qubits)
    new_row = np.zeros(n_qubits + 1)
    # x destabilizer part

    tmp_dex = np.insert(tableau.destabilizer_x, new_qubit_position, new_column, axis=1)
    tmp_dex = np.insert(tmp_dex, new_qubit_position, new_row, axis=0)

    # z destabilizer part
    tmp_dez = np.insert(tableau.destabilizer_z, new_qubit_position, new_column, axis=1)
    tmp_dez = np.insert(tmp_dez, new_qubit_position, new_row, axis=0)

    # x stabilizer part
    tmp_sx = np.insert(tableau.stabilizer_x, new_qubit_position, new_column, axis=1)
    tmp_sx = np.insert(tmp_sx, new_qubit_position, new_row, axis=0)

    # z stabilizer part
    tmp_sz = np.insert(tableau.stabilizer_z, new_qubit_position, new_column, axis=1)
    tmp_sz = np.insert(tmp_sz, new_qubit_position, new_row, axis=0)

    # phase vector part
    new_phase = np.insert(
        tableau.phase, [new_qubit_position, n_qubits + 1 + new_qubit_position], 0
    )

    new_table = np.block([[tmp_dex, tmp_dez], [tmp_sx, tmp_sz]])
    tableau.expand(new_table, new_phase)

    # set the new qubit to ket 0 state
    tableau.destabilizer_x[new_qubit_position, new_qubit_position] = 1
    tableau.stabilizer_z[new_qubit_position, new_qubit_position] = 1

    return tableau


def remove_qubit(tableau, qubit_position, measurement_determinism="probabilistic"):
    """
    The action of the function is measure and remove. If isolated, state should not change. Entangled qubits cannot be
    removed without affecting other parts of the state.
    Only works correctly for isolated qubits! e.g. after measurement.
    TODO: Check if a qubit is isolated in general? Only isolated qubits in the Z basis states can be confirmed for now.

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
    new_phase = np.delete(tableau.phase, [qubit_position, qubit_position + n_qubits])
    if probabilistic:
        new_table = np.delete(
            new_table, [probabilistic, probabilistic - n_qubits], axis=0
        )

    else:
        non_zero = [
            i for i in range(n_qubits) if tableau.destabilizer_x[i, qubit_position] != 0
        ]
        if len(non_zero) <= 1:
            new_table = np.delete(
                new_table, [non_zero[0], non_zero[0] + n_qubits], axis=0
            )

        else:
            for i in non_zero:
                if np.array_equal(new_table[i], np.zeros(2 * n_qubits - 2)):
                    new_table = np.delete(new_table, [i, i + n_qubits], axis=0)

            new_table = np.delete(
                new_table, [non_zero[-1], non_zero[-1] + n_qubits], axis=0
            )

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

    :return: the classical outcome of measuring given qubit in the X basis.
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
    Returns the outcome 0 or 1 if one measures the given qubit in the X basis.
    NOTE: Does not update the stabilizer state after measurement.

    :param tableau:
    :type tableau: CliffordTableau
    :param qubit_position: index of the qubit that the gate acts on
    :type qubit_position: int
    :param measurement_determinism:
    :type measurement_determinism:
    :return: the classical outcome of measuring given qubit in the X basis.
    :rtype: int
    """
    _, outcome, _ = z_measurement_gate(tableau, qubit_position, measurement_determinism)
    return outcome


def swap_gate(tableau, qubit1, qubit2):
    """
    Swap gate between two qubits.

    :param tableau:
    :param qubit1: One of the qubits as input to the swap gate. (symmetrical)
    :param qubit2: The other qubit.
    :return: Updated state
    """
    n_qubits = tableau.n_qubits  # number of qubits
    assert qubit1 < n_qubits and qubit2 < n_qubits
    tableau.table = column_swap(tableau.table, qubit1, qubit2)
    tableau.table = column_swap(tableau.table, qubit1 + n_qubits, qubit2 + n_qubits)
    return tableau


def create_n_ket0_state(n_qubits):
    """
    Creates a product state that consists :math:`n` tensor factors of the computational (Pauli Z) 0 state.
    """

    return CliffordTableau(n_qubits)


def create_n_plus_state(n_qubits):
    """
    Creates a product state that consists :math:`n` tensor factors of the "plus state" (Pauli X's +1 eigenstate)
    """
    tableau = create_n_ket0_state(n_qubits)

    tableau.table[:, [*range(2 * n_qubits)]] = tableau.table[
        :, [*range(n_qubits, 2 * n_qubits)] + [*range(0, n_qubits)]
    ]

    return tableau


def tensor(list_of_tables):
    """

    :param list_of_tables: A list of the states' tableau we want to tensor in the same order. (left to right)
    :type list_of_tables: list[CliffordTableau]
    :return: The resulting tableau
    :rtype: CliffordTableau
    """
    tableau = list_of_tables[0]
    list_of_tables = list_of_tables[1:]
    for tab in list_of_tables:
        tableau.destabilizer_x = block_diag(tableau.destabilizer_x, tab.destabilizer_x)
        tableau.destabilizer_z = block_diag(tableau.destabilizer_z, tab.destabilizer_z)
        tableau.stabilizer_x = block_diag(tableau.stabilizer_x, tab.stabilizer_x)
        tableau.stabilizer_z = block_diag(tableau.stabilizer_z, tab.stabilizer_z)
        phase_list1 = np.split(tableau.phase, 2)
        phase_list2 = np.split(tab.phase, 2)
        phase_vector = np.hstack(
            (phase_list1[0], phase_list2[0], phase_list1[1], phase_list2[1])
        )
        tableau.phase = phase_vector
    return tableau


def partial_trace():
    """
    TODO: Need to use this to remove emitter qubits.

    :return:
    :rtype:
    """
    pass


def fidelity(tableau1, tableau2):
    """
    Compute the fidelity of two stabilizer states given their tableaux.

    :param tableau1:
    :type tableau1:
    :param tableau2:
    :type tableau2:
    :return:
    :rtype: float
    """
    return np.abs(inner_product(tableau1, tableau2)) ** 2


def full_rank_x(tableau):
    #Based on lemma 6 in arXiv:quant-ph/0406196
    n_qubits = tableau.n_qubits
    x_matrix = tableau.stabilizer_x
    z_matrix = tableau.stabilizer_x
    x_matrix, z_matrix, index = row_reduction(x_matrix, z_matrix)
    rank = index + 1
    z1_matrix = z_matrix[rank:n_qubits, 0:rank]
    z_2matrix = z_matrix[rank:n_qubits, rank:n_qubits]
    z_2matrix, z1_matrix, index1 = row_reduction(z_2matrix, z1_matrix)
    assert index1 == n_qubits - rank - 1
    for j in range(n_qubits - rank):
        for i in range(j):
            if z_2matrix[i, j] == 1:
                z_2matrix = add_rows(z_2matrix, j, i)
                z_1matrix = add_rows(z_1matrix, j, i)

    assert np.array_equal(z_2matrix, np.eye(n_qubits - rank))
    z_matrix = np.hstack(z_1matrix, z_2matrix)
    tableau.x_stabilizer = x_matrix
    tableau.z_stabilizer = z_matrix
    # hadamard on some qubits to make the x-stabilizer table full rank
    for qubit in range(rank, n_qubits):
        tableau = hadamard_gate(tableau, qubit)

    return tableau


def trace_out():
    pass


def set_qubit(qubit, intended_state):
    """probably no possible to implement"""
    pass


def project_to_z0_and_remove(tableau, locations):
    # probably not implementing this
    pass


def inverse_circuit(tableau):
    """
    TODO: implement this function

    :param tableau:
    :type tableau:
    :return:
    :rtype:
    """

    # apply Hadamard gates to make the stabilizer_x full rank

    # apply CNOT gates to perform Gaussian elimination on stabilizer_x

    # apply phase gates to make stabilizer_z full rank

    # find invertible matrix M

    # apply CNOT gates to make stabilizer (M, M)

    # apply phase gates to make stabilizer_x zero matrix

    # apply CNOT gates to perform Gaussian elimination on stabilizer_x

    # apply Hadamard gates to make destabilizer_x = identity,
    # destabilizer_z = previous destabilizer_x
    # stabilizer_x = zero matrix,
    # destabilizer_z = identity

    # apply phase gates to make destabilizer_z invertible

    # find the invertible matrix N

    # apply CNOT gates to make destabilizer_x = N,
    # destabilizer_z = N, stabilizer_x = 0,
    # stabilizer_z = original stabilizer_x

    # apply phase gates to make destabilizer_z = 0

    # apply CNOT gates to get the tableau of all ket 0 states

    return


def min_generator_distance(tableau1, tableau2):
    """
    TODO: implement this function.

    :param tableau1:
    :type tableau1:
    :param tableau2:
    :type tableau2:
    :return:
    :rtype:
    """
    # unitary = inverse_circuit(tableau1)

    # apply the unitary to the state tableau2
    tableau2_transformed = tableau2  # TODO: update this line

    stabilizer1 = tableau1.stabilizer
    stabilizer2 = tableau2_transformed.stabilizer
    counter = 0
    for i in range(stabilizer1.shape[0]):
        if ~np.allclose(stabilizer1[i], stabilizer2[i]):
            counter += 1
    return counter


def inner_product(tableau1, tableau2):
    """
    TODO: implement this function

    :param tableau1:
    :type tableau1:
    :param tableau2:
    :type tableau2:
    :return:
    :rtype:
    """

    return 2 ** (-min_generator_distance(tableau1, tableau2) / 2)
