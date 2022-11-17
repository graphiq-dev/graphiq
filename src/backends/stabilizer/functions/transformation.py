"""
The gate transformation here is applicable to both StabilizerTableau and CliffordTableau.
"""

from src.backends.stabilizer.functions.linalg import (
    multiply_columns,
    column_swap,
    add_columns,
)

"""Main gates """


def identity(tableau, *args):
    """
    Apply the identity operation on a tableau, i.e., does not change the state at all.

    :param tableau: Tableau of the state before gate action
    :type tableau: CliffordTableau or StabilizerTableau
    :return: the identical state
    :rtype: CliffordTableau or StabilizerTableau
    """
    return tableau


def hadamard_gate(tableau, qubit_position):
    """
    Apply Hadamard gate to a single qubit given its position in a stabilizer state.

    :param tableau: Tableau of the state before gate action
    :type tableau: CliffordTableau or StabilizerTableau
    :param qubit_position: index of the qubit that the gate acts on
    :type qubit_position: int
    :return: the resulting state after gate action
    :rtype: CliffordTableau or StabilizerTableau
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
    Apply phase gate to the qubit given its position in a stabilizer state.

    :param tableau: Tableau of the state before gate action
    :type tableau: CliffordTableau or StabilizerTableau
    :param qubit_position: index of the qubit that the gate acts on
    :type qubit_position: int
    :return: the resulting state after gate action
    :rtype: CliffordTableau or StabilizerTableau
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
    Apply CNOT to control and target qubits given their position in a stabilizer state.

    :param tableau: Tableau of the state before gate action
    :type tableau: CliffordTableau or StabilizerTableau
    :param ctrl_qubit: index of the control qubit
    :type ctrl_qubit: int
    :param target_qubit: index of the target qubit that the gate acts on
    :type target_qubit: int
    :return: the resulting state after gate action
    :rtype: CliffordTableau or StabilizerTableau
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


"""SECONDARY GATES """


def phase_dagger_gate(tableau, qubit_position):
    """
    Apply Phase dagger gate (inverse of phase) to a given qubit in a stabilizer state tableau.

    :param tableau: Tableau of the state before gate action
    :type tableau: CliffordTableau or StabilizerTableau
    :param qubit_position: index of the qubit that the gate acts on
    :type qubit_position: int
    :return: the resulting state after gate action
    :rtype: CliffordTableau or StabilizerTableau
    """

    for _ in range(3):
        tableau = phase_gate(tableau, qubit_position)

    return tableau


def z_gate(tableau, qubit_position):
    """
    Apply the Pauli Z to a given qubit in a stabilizer state tableau.

    :param tableau: Tableau of the state before gate action
    :type tableau: CliffordTableau or StabilizerTableau
    :param qubit_position: index of the qubit that the gate acts on
    :type qubit_position: int
    :return: the resulting state after gate action
    :rtype: CliffordTableau or StabilizerTableau
    """

    for _ in range(2):
        tableau = phase_gate(tableau, qubit_position)

    return tableau


def x_gate(tableau, qubit_position):
    """
    Apply the Pauli X (= HZH) to a given qubit in a stabilizer state tableau.

    :param tableau: Tableau of the state before gate action
    :type tableau: CliffordTableau or StabilizerTableau
    :param qubit_position: index of the qubit that the gate acts on
    :type qubit_position: int
    :return: the resulting state after gate action
    :rtype: CliffordTableau or StabilizerTableau
    """

    tableau = hadamard_gate(tableau, qubit_position)
    tableau = z_gate(tableau, qubit_position)
    tableau = hadamard_gate(tableau, qubit_position)

    return tableau


def y_gate(tableau, qubit_position):
    """
    Apply the Pauli Y (=PXZP) to a given qubit in a stabilizer state tableau.

    :param tableau: Tableau of the state before gate action
    :type tableau: CliffordTableau or StabilizerTableau
    :param qubit_position: index of the qubit that the gate acts on
    :type qubit_position: int
    :return: the resulting state after gate action
    :rtype: CliffordTableau or StabilizerTableau
    """

    tableau = phase_gate(tableau, qubit_position)
    tableau = z_gate(tableau, qubit_position)
    tableau = x_gate(tableau, qubit_position)
    tableau = phase_gate(tableau, qubit_position)

    return tableau


def control_z_gate(tableau, ctrl_qubit, target_qubit):
    """
    Apply controlled Z gate to control and target qubits in a stabilizer state tableau.

    :param tableau: Tableau of the state before gate action
    :type tableau: CliffordTableau or StabilizerTableau
    :param ctrl_qubit: index of the control qubit
    :type ctrl_qubit: int
    :param target_qubit: index of the target qubit that the gate acts on
    :type target_qubit: int
    :return: the resulting state after gate action
    :rtype: CliffordTableau or StabilizerTableau
    """
    tableau = hadamard_gate(tableau, target_qubit)
    tableau = cnot_gate(tableau, ctrl_qubit, target_qubit)
    tableau = hadamard_gate(tableau, target_qubit)
    return tableau


def control_y_gate(tableau, ctrl_qubit, target_qubit):
    """
    Apply controlled Y gate to control and target qubits in a stabilizer state tableau.

    :param tableau: Tableau of the state before gate action
    :type tableau: CliffordTableau or StabilizerTableau
    :param ctrl_qubit: index of the control qubit
    :type ctrl_qubit: int
    :param target_qubit: index of the target qubit that the gate acts on
    :type target_qubit: int
    :return: the resulting state after gate action
    :rtype: CliffordTableau or StabilizerTableau
    """
    tableau = phase_gate(tableau, target_qubit)
    tableau = z_gate(tableau, target_qubit)
    tableau = cnot_gate(tableau, ctrl_qubit, target_qubit)
    tableau = phase_gate(tableau, target_qubit)
    return tableau


def run_circuit(tableau, circuit_list, reverse=False):
    """
    Return the stabilizer state tableau after the execution of the circuit.

    :param tableau: initial state tableau
    :type tableau: CliffordTableau or StabilizerTableau
    :param circuit_list: a list of gates in the circuit
    :type circuit_list: list[tuple]
    :param reverse: a parameter to indicate whether running the inverse circuit
    :type reverse: bool
    :return: the stabilizer state tableau after the execution of the circuit.
    :rtype: CliffordTableau or StabilizerTableau
    """
    if reverse:
        circuit_list.reverse()
    for ops in circuit_list:
        if ops[0] == "H":
            tableau = hadamard_gate(tableau, ops[1])
        elif ops[0] == "P":
            if reverse:
                tableau = phase_dagger_gate(tableau, ops[1])
            else:
                tableau = phase_gate(tableau, ops[1])
        elif ops[0] == "X":
            tableau = x_gate(tableau, ops[1])
        elif ops[0] == "Y":
            tableau = y_gate(tableau, ops[1])
        elif ops[0] == "Z":
            tableau = z_gate(tableau, ops[1])
        elif ops[0] == "CNOT":
            tableau = cnot_gate(tableau, ops[1], ops[2])
        elif ops[0] == "CZ":
            tableau = control_z_gate(tableau, ops[1], ops[2])
    return tableau
