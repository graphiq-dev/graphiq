import numpy as np
from src.backends.stabilizer.tableau import CliffordTableau
from src.backends.stabilizer.functions.matrix_functions import (
    multiply_columns,
    column_swap,
    add_columns,
    row_sum,
)


### Main gates
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
        tableau, tableau, qubit_position, n_qubits + qubit_position
    )
    # updating the rest of the tableau
    tableau.table = column_swap(
        tableau.table, qubit_position, n_qubits + qubit_position
    )

    return tableau


def phase_gate(tableau, qubit_position):
    """
    phase gate applied on a single qubit given its position, in a stabilizer state.

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


def z_measurement_gate(tableau, qubit_position, measurement_determinism="probabilistic"):
    """
    Measurement applied on a single qubit given its position, in a stabilizer state.

    :param tableau:
    :type tableau: CliffordTableau
    :param qubit_position: index of the qubit that the gate acts on
    :type qubit_position: int
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
    for non_zero in non_zero_x:
        if non_zero >= n_qubits:
            x_p = non_zero
            non_zero_x = np.delete(non_zero_x, x_p)
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

        table[x_p] = np.zeros(2 * n_qubits + 1)
        table[x_p, qubit_position + n_qubits] = 1
        # set r_vector of that row to random measurement outcome 0 or 1. (equal probability)
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
        # add an extra 2n+1 th row to the tableau
        new_table = np.vstack([tableau.table, np.zeros(2 * n_qubits)])
        x_matrix = new_table[:, 0:n_qubits]
        z_matrix = new_table[:, n_qubits: 2 * n_qubits]
        r_vector = np.vstack([tableau.phase, np.zeros(1)])

        # list of nonzero elements in the x destabilizers
        non_zero_x = [i for i in non_zero_x if i < n_qubits]
        for non_zero in non_zero_x:
            x_matrix, z_matrix, r_vector = row_sum(
                x_matrix, z_matrix, r_vector, non_zero + n_qubits, 2 * n_qubits
            )
        # update the tableau?
        tableau.table_x = x_matrix[:-1, :]
        tableau.table_z = z_matrix[:-1, :]
        tableau.phase = r_vector[:-1]
        outcome = r_vector[2 * n_qubits]
        return tableau, outcome, x_p


### SECONDARY GATES
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
    Pauli Z applied on a single qubit given its position, in a stabilizer state.

    :param tableau:
    :type tableau: CliffordTableau
    :param qubit_position: index of the qubit that the gate acts on
    :type qubit_position: int
    :return: the resulting state after gate action
    :rtype: StabilizerState
    """
    n_qubits = tableau.n_qubits  # number of qubits
    assert qubit_position < n_qubits
    for _ in range(2):
        tableau = phase_gate(tableau, qubit_position)

    return tableau


def x_gate(tableau, qubit_position):
    """
    Pauli X (= HZH) applied on a single qubit given its position, in a stabilizer state.

    :param tableau:
    :type tableau: CliffordTableau
    :param qubit_position: index of the qubit that the gate acts on
    :type qubit_position: int
    :return: the resulting state after gate action
    :rtype: StabilizerState
    """
    n_qubits = tableau.n_qubits  # number of qubits
    assert qubit_position < n_qubits
    tableau = hadamard_gate(tableau, qubit_position)
    tableau = z_gate(tableau, qubit_position)
    tableau = hadamard_gate(tableau, qubit_position)

    return tableau


def y_gate(tableau, qubit_position):
    """
    Pauli Y (=PXZP) applied on a single qubit given its position, in a stabilizer state.

    :param tableau:
    :type tableau: CliffordTableau
    :param qubit_position: index of the qubit that the gate acts on
    :type qubit_position: int
    :return: the resulting state after gate action
    :rtype: StabilizerState
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
    :rtype: StabilizerState
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
    :rtype: StabilizerState
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
    :rtype: StabilizerState
    """
    tableau = phase_gate(tableau, target_qubit)
    tableau = z_gate(tableau, target_qubit)
    tableau = cnot_gate(tableau, ctrl_qubit, target_qubit)
    tableau = phase_gate(tableau, target_qubit)
    return tableau


def projector_z0(tableau, qubit_position, measurement_determinism="probabilistic"):
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
            tableau.table[probabilistic, -1] = 0
    else:
        if outcome == 0:
            pass
        else:
            tableau = 0
            success = False
            # TODO: see how impossible projection is handled in the density matrix formalism
    return tableau, success


def projector_z1(tableau, qubit_position, measurement_determinism="probabilistic"):
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
            tableau.table[probabilistic, -1] = 1
    else:
        if outcome == 1:
            pass
        else:
            tableau = 0
            success = False
            # TODO: see how impossible projection is handled in the density matrix formalism
    return tableau, success


def reset_qubit(
        tableau, qubit_position, intended_state, measurement_determinism="probabilistic"
):
    # reset qubit to computational basis states
    # NOTE: Only works after a measurement gate on the same qubit or for isolated qubits.
    n_qubits = tableau.n_qubits  # number of qubits
    assert qubit_position < n_qubits
    assert intended_state == 0 or intended_state == 1
    tableau, outcome, probabilistic = z_measurement_gate(tableau, qubit_position, measurement_determinism)
    if probabilistic:
        if outcome == intended_state:
            pass
        else:
            tableau.table[probabilistic, -1] = intended_state
    else:
        if outcome == intended_state:
            pass
        else:
            the_generator = np.zeros(2 * n_qubits)
            the_generator[n_qubits + qubit_position] = 1

            generators = tableau.stabilizer
            for i in range(n_qubits):
                if generators[i] == the_generator:
                    assert tableau.table[i, qubit_position] == 1, (
                        "unexpected destabilizer element. The "
                        "reset gate in probably not used after a"
                        " valid measurement on the same qubit "
                    )
                    tableau.phase[i] = 1 ^ tableau.phase[i]
    return tableau


def set_qubit(qubit, intended_state):
    """probably no possible to implement"""
    pass


def add_qubit(tableau):
    """
    add one isolated qubit in 0 state of the computational basis to the current state.

    :return: the updated stabilizer state
    """
    n_qubits = tableau.n_qubits  # number of qubits
    n_new = 1 + n_qubits
    if n_qubits == 0:
        return create_n_product_state(1)

    new_tableau = create_n_product_state(n_new)
    # x destabilizer part
    new_tableau.destabilizer_x[0:n_qubits,0:n_qubits] = tableau.destabilizer_x
    # z destabilizer part
    new_tableau.destabilizer_z[0:n_qubits,0:n_qubits] = tableau.destabilizer_z
    # r destabilizer part
    new_tableau.phase[0:n_qubits] = tableau.phase[0, n_qubits]
    # x stabilizer part
    new_tableau.stabilizer_x[0:n_qubits, 0:n_qubits] = tableau.stabilizer_x
    # z stabilizer part
    new_tableau.stabilizer_z[0:n_qubits, 0:n_qubits] = tableau.stabilizer_z
    # r stabilizer part
    new_tableau.phase[n_new: 2 * n_new - 1] = tableau.phase[n_qubits: 2 * n_qubits]

    tableau.table = new_tableau.table
    tableau.phase = tableau.phase
    tableau.n_qubits = n_new
    return new_tableau


def insert_qubit(tableau, insert_after):
    """
    To be implemented. Similar to `add qubit` function.
    :param tableau:
    :param insert_after: the position index after which a new qubit is added.
    :return: updated state
    """


def remove_qubit(tableau, qubit_position, measurement_determinism="probabilistic"):
    """
    Needs further investigation. Code below only works for isolated qubits.
    #TODO: Check if a qubit is isolated in general? Only isolated qubits in the Z basis states can be confirmed for now.
    The action of the function is measure and remove. If isolated, state should not change.
    only works correctly for isolated qubits! e.g. after measurement
    entangled qubits cannot be removed without affecting other parts of the state

    NEW method: discard any of the eligible rows from the stabilizer and destabilizer sets instead of finding THE one.
    """

    new_tableau, _, probabilistic = z_measurement_gate(
        tableau, qubit_position, measurement_determinism
    )
    if probabilistic:
        row_position = probabilistic
    else:
        pass
    # number of qubits
    n_qubits = tableau.n_qubits
    assert qubit_position < n_qubits
    table = tableau.table
    table = np.delete(table, qubit_position, axis=1)
    table = np.delete(table, n_qubits + qubit_position, axis=1)
    row_positions = [i for i in range(len(tableau)) if not np.any(table[i])]
    assert len(row_positions) == 1
    row_position = row_positions[0]
    # order of removing rows matters here
    table = np.delete(table, n_qubits + row_position)
    table = np.delete(table, row_position)
    tableau.table = table

    return tableau


def measure_x(tableau, qubit_position, measurement_determinism="probabilistic"):
    """
    Returns the outcome 0 or 1 if one measures the given qubit in the X basis.
    NOTE: cannot update the stabilizer state after measurement. Stabilizer formalism can only handle Z-measurements.

    :param tableau: a StabilizerState object.
    :type tableau: StabilizerState
    :param qubit_position: index of the qubit that the gate acts on
    :type qubit_position: int
    :param seed: a seed for random outcome of the measurement
    :type seed: int
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

    :param tableau: a StabilizerState object.
    :type tableau: StabilizerState
    :param qubit_position: index of the qubit that the gate acts on
    :type qubit_position: int
    :param measurement_determinism:
    :type measurement_determinism:
    :return: the classical outcome of measuring given qubit in the X basis.
    :rtype: int
    """
    _, outcome, _ = z_measurement_gate(tableau, qubit_position, measurement_determinism)
    return outcome


def trace_out():
    pass


def swap_gate(tableau, qubit1, qubit2):
    pass


def create_n_product_state(n_qubits):
    """
    Creates a product state that consists :math:`n` tensor factors of the computational (Pauli Z) 0 state.
    """

    return CliffordTableau(n_qubits)


def create_n_plus_state(n_qubits):
    """
    Creates a product state that consists :math:`n` tensor factors of the "plus state" (Pauli X's +1 eigenstate)
    """
    tableau = create_n_product_state(n_qubits)

    tableau.table[:, [*range(2 * n_qubits)]] = tableau.table[
                                               :, [*range(n_qubits, 2 * n_qubits)] + [*range(0, n_qubits)]
                                               ]

    return tableau


def tensor(list_of_states):
    pass


def partial_trace():
    pass


def fidelity(a, b):
    pass


def project_to_z0_and_remove(tableau, locations):
    # what if it cannot be projected to z0 ?! how is it
    # handled in density matrix formalism? ask
    pass
