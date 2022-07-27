import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
import random
import src.backends.stabilizer.state as ss
import src.backends.density_matrix.functions as dmf


def symplectic_to_string(x_matrix, z_matrix):
    """
    Convert a binary symplectic representation to a list of strings

    :param x_matrix: X part of the binary symplectic representation
    :param z_matrix: Z part of the binary symplectic representation
    :return: a list of strings that represent stabilizer generators
    :rtype: list[str]
    """
    assert x_matrix.shape == z_matrix.shape
    n_row, n_column = x_matrix.shape
    generator_list = []
    for i in range(n_row):
        generator = ""
        for j in range(n_column):
            if x_matrix[i, j] == 1 and z_matrix[i, j] == 0:
                generator = generator + "X"
            elif x_matrix[i, j] == 1 and z_matrix[i, j] == 1:
                generator = generator + "Y"
            elif x_matrix[i, j] == 0 and z_matrix[i, j] == 1:
                generator = generator + "Z"
            else:
                generator = generator + "I"
        generator_list.append(generator)
    return generator_list


def string_to_symplectic(generator_list):
    """
    Convert a string list representation of stabilizer generators to a symplectic representation

    :param generator_list: a list of strings
    :type generator_list: list[str]
    :return: two binary matrices, one for X part, the other for Z part
    :rtype: numpy.ndarray, numpy.ndarray
    """
    n_row = len(generator_list)
    n_column = len(generator_list[0])
    x_matrix = np.zeros((n_row, n_column))
    z_matrix = np.zeros((n_row, n_column))
    for i in range(n_row):
        generator = generator_list[i]
        for j in range(n_column):
            if generator[j].lower() == "x":
                x_matrix[i, j] = 1
            elif generator[j].lower() == "y":
                x_matrix[i, j] = 1
                z_matrix[i, j] = 1
            elif generator[j].lower() == "z":
                z_matrix[i, j] = 1
    return x_matrix, z_matrix


def row_swap(input_matrix, first_row, second_row):
    """
    Swap two rows of a matrix

    :param input_matrix: a matrix
    :type input_matrix: numpy.ndarray
    :param first_row: the first row
    :type first_row: int
    :param second_row: the second row
    :type second_row: int
    :return: the matrix after swapping those two row
    :rtype: numpy.ndarray
    """
    input_matrix[[first_row, second_row]] = input_matrix[[second_row, first_row]]
    return input_matrix


def add_rows(input_matrix, row_to_add, resulting_row):
    """
    Add two rows together modulo 2 and put it in the row of the second input

    :param input_matrix: a binary matrix
    :type input_matrix: numpy.ndarray
    :param row_to_add: the index of the row to add
    :type row_to_add: int
    :param resulting_row: the index of the row where the result is put
    :type resulting_row: int
    :return: the matrix after adding two rows modulo 2 and putting in the row of the second input
    :rtype: numpy.ndarray
    """
    input_matrix[resulting_row] = (
        input_matrix[row_to_add] + input_matrix[resulting_row]
    ) % 2
    return input_matrix


def column_swap(input_matrix, first_col, second_col):
    """
    Swap two columns of a matrix

    :param input_matrix: a matrix
    :type input_matrix: numpy.ndarray
    :param first_col: the first column
    :type first_col: int
    :param second_col: the second column
    :type second_col: int
    :return: the matrix after swapping those two columns
    :rtype: numpy.ndarray
    """
    input_matrix[:, [first_col, second_col]] = input_matrix[:, [second_col, first_col]]
    return input_matrix


def add_columns(input_matrix, col_to_add, resulting_col):
    """
    Add two rows together modulo 2 and put it in the row of the second input

    :param input_matrix: a binary matrix
    :type input_matrix: numpy.ndarray
    :param col_to_add: the index of the column to add
    :type col_to_add: int
    :param resulting_col: the index of the column where the result is put
    :type resulting_col: int
    :return: the matrix after adding two column modulo 2 and putting in the column of the second input
    :rtype: numpy.ndarray
    """
    input_matrix[:, resulting_col] = (
        input_matrix[:, col_to_add] + input_matrix[:, resulting_col]
    ) % 2
    return input_matrix


def multiply_columns(matrix_one, matrix_two, first_col, second_col):
    """
    Multiplies two columns of possibly two matrices (element-wise), and returns a column containing the result.

    :param matrix_one: a matrix
    :type matrix_one: numpy.ndarray
    :param matrix_two: a second matrix of the same number of rows as the first one
    :type matrix_two: numpy.ndarray
    :param first_col: index of the column to be used from the first matrix
    :type first_col: int
    :param second_col: index of the column to be used from the second matrix
    :type second_col: int
    :return: the resulting 1-d array of length n (= number of the rows of the matrices)
    :rtype: numpy.ndarray
    """
    n_rows, _ = np.shape(matrix_one)
    assert np.shape(matrix_one)[0] == np.shape(matrix_two)[0]
    try:
        assert (
            first_col < np.shape(matrix_one)[1] and second_col < np.shape(matrix_two)[1]
        )
    except:
        raise ValueError(
            "the specified column index is out of range in one of the matrices"
        )
    resulting_col = np.multiply(matrix_one[:, first_col], matrix_two[:, second_col])
    # reshape into column form:
    # resulting_col = resulting_col.reshape(n_rows, 1)
    return resulting_col


### MAIN GATES:
def hadamard_gate(stabilizer_state, qubit_position):
    """
    hadamard gate applied on a single qubit given its position, in a stabilizer state.

    :param stabilizer_state: a StabilizerState object.
    :type stabilizer_state: StabilizerState
    :param qubit_position: index of the qubit that the gate acts on
    :type qubit_position: int
    :return: the resulting state after gate action
    :rtype: StabilizerState
    """
    n_qubits = stabilizer_state.n_qubits  # number of qubits
    assert qubit_position < n_qubits
    tableau = stabilizer_state.tableau  # full tableau as a np matrix n*(2n+1)
    # updating phase vector
    tableau[:, -1] = tableau[:, -1] ^ multiply_columns(
        tableau, tableau, qubit_position, n_qubits + qubit_position
    )
    # updating the rest of the tableau
    tableau = column_swap(tableau, qubit_position, n_qubits + qubit_position)
    stabilizer_state.tableau = tableau
    return stabilizer_state


def phase_gate(stabilizer_state, qubit_position):
    """
    phase gate applied on a single qubit given its position, in a stabilizer state.

    :param stabilizer_state: a StabilizerState object.
    :type stabilizer_state: StabilizerState
    :param qubit_position: index of the qubit that the gate acts on
    :type qubit_position: int
    :return: the resulting state after gate action
    :rtype: StabilizerState
    """
    n_qubits = stabilizer_state.n_qubits  # number of qubits
    assert qubit_position < n_qubits
    tableau = stabilizer_state.tableau  # full tableau as a np matrix n*(2n+1)
    # updating phase vector
    tableau[:, -1] = tableau[:, -1] ^ multiply_columns(
        tableau, tableau, qubit_position, n_qubits + qubit_position
    )
    # updating the rest of the tableau
    tableau = add_columns(tableau, qubit_position, n_qubits + qubit_position)
    stabilizer_state.tableau = tableau
    return stabilizer_state


def cnot_gate(stabilizer_state, ctrl_qubit, target_qubit):
    """
    CNOT on control and target qubits given their position, in a stabilizer state.

    :param stabilizer_state: a StabilizerState object.
    :type stabilizer_state: StabilizerState
    :param ctrl_qubit: index of the control qubit
    :type ctrl_qubit: int
    :param target_qubit: index of the target qubit that the gate acts on
    :type target_qubit: int
    :return: the resulting state after gate action
    :rtype: StabilizerState
    """
    n_qubits = stabilizer_state.n_qubits  # number of qubits
    assert ctrl_qubit < n_qubits and target_qubit < n_qubits
    tableau = stabilizer_state.tableau  # full tableau as a np matrix n*(2n+1)
    # updating phase vector
    x_ctrl_times_z_target = multiply_columns(
        tableau, tableau, ctrl_qubit, n_qubits + target_qubit
    )
    x_target = tableau[:, target_qubit]
    z_ctrl = tableau[:, n_qubits + ctrl_qubit]
    tableau[:, -1] = tableau[:, -1] ^ (x_ctrl_times_z_target * (x_target ^ z_ctrl ^ 1))
    # updating the rest of the tableau
    tableau[:, target_qubit] = tableau[:, target_qubit] ^ tableau[:, ctrl_qubit]
    tableau[:, n_qubits + ctrl_qubit] = (
        tableau[:, n_qubits + ctrl_qubit] ^ tableau[:, n_qubits + target_qubit]
    )
    stabilizer_state.tableau = tableau
    return stabilizer_state


def z_measurement_gate(stabilizer_state, qubit_position, seed=0):
    """
    Measurement applied on a single qubit given its position, in a stabilizer state.

    :param stabilizer_state: a StabilizerState object.
    :type stabilizer_state: StabilizerState
    :param qubit_position: index of the qubit that the gate acts on
    :type qubit_position: int
    :param seed: a seed for random outcome of the measurement
    :type seed: int
    :return: the resulting state after gate action, the measurement outcome, whether the measurement outcome is
    probabilistic (zero means deterministic)
    :rtype: StabilizerState, int, int
    """
    n_qubits = stabilizer_state.n_qubits  # number of qubits
    assert qubit_position < n_qubits
    tableau = stabilizer_state.tableau  # full tableau as a np matrix n*(2n+1)
    x_column = tableau[:, qubit_position]
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
        x_matrix = tableau[:, 0:n_qubits]
        z_matrix = tableau[:, n_qubits : 2 * n_qubits]
        r_vector = tableau[:, -1]
        # rowsum for all other x elements equal to 1 other than x_p
        for target_row in non_zero_x:
            x_matrix, z_matrix, r_vector = row_sum(
                x_matrix, z_matrix, r_vector, x_p, target_row
            )
        tableau[:, 0:n_qubits] = x_matrix
        tableau[:, n_qubits : 2 * n_qubits] = z_matrix
        tableau[:, -1] = r_vector
        # set x_p - n row equal to x_p row
        tableau[x_p - n_qubits] = tableau[x_p]
        # set x_p row equal to 0 except for z element of measured qubit which is 1.
        tableau[x_p] = np.zeros(2 * n_qubits + 1)
        tableau[x_p, qubit_position + n_qubits] = 1
        # set r_vector of that row to random measurement outcome 0 or 1. (equal probability)
        random.seed(seed)
        tableau[x_p, -1] = random.randint(0, 1)

        stabilizer_state.tableau = tableau
        return stabilizer_state, tableau[x_p, 1 + 2 * n_qubits], x_p
    else:
        # deterministic outcome
        # add an extra 2n+1 th row to the tableau
        new_tableau = np.vstack([tableau, np.zeros(1 + 2 * n_qubits)])
        x_matrix = new_tableau[:, 0:n_qubits]
        z_matrix = new_tableau[:, n_qubits : 2 * n_qubits]
        r_vector = new_tableau[:, -1]

        non_zero_x = [
            i for i in non_zero_x if i < n_qubits
        ]  # list of nonzero elements in the x destabilizers
        for non_zero in non_zero_x:
            x_matrix, z_matrix, r_vector = row_sum(
                x_matrix, z_matrix, r_vector, non_zero + n_qubits, 2 * n_qubits
            )
        return stabilizer_state, r_vector[2 * n_qubits], x_p


### SECONDARY GATES
def phase_dagger_gate(stabilizer_state, qubit_position):
    """
    Phase dagger gate (inverse of phase) applied on a single qubit given its position, in a stabilizer state.

    :param stabilizer_state: a StabilizerState object.
    :type stabilizer_state: StabilizerState
    :param qubit_position: index of the qubit that the gate acts on
    :type qubit_position: int
    :return: the resulting state after gate action
    :rtype: StabilizerState
    """
    n_qubits = stabilizer_state.n_qubits  # number of qubits
    assert qubit_position < n_qubits
    for i in range(3):
        stabilizer_state = phase_gate(stabilizer_state, qubit_position)

    return stabilizer_state


def z_gate(stabilizer_state, qubit_position):
    """
    Pauli Z applied on a single qubit given its position, in a stabilizer state.

    :param stabilizer_state: a StabilizerState object.
    :type stabilizer_state: StabilizerState
    :param qubit_position: index of the qubit that the gate acts on
    :type qubit_position: int
    :return: the resulting state after gate action
    :rtype: StabilizerState
    """
    n_qubits = stabilizer_state.n_qubits  # number of qubits
    assert qubit_position < n_qubits
    for i in range(2):
        stabilizer_state = phase_gate(stabilizer_state, qubit_position)

    return stabilizer_state


def x_gate(stabilizer_state, qubit_position):
    """
    Pauli X (= HZH) applied on a single qubit given its position, in a stabilizer state.

    :param stabilizer_state: a StabilizerState object.
    :type stabilizer_state: StabilizerState
    :param qubit_position: index of the qubit that the gate acts on
    :type qubit_position: int
    :return: the resulting state after gate action
    :rtype: StabilizerState
    """
    n_qubits = stabilizer_state.n_qubits  # number of qubits
    assert qubit_position < n_qubits
    stabilizer_state = hadamard_gate(stabilizer_state, qubit_position)
    stabilizer_state = z_gate(stabilizer_state, qubit_position)
    stabilizer_state = hadamard_gate(stabilizer_state, qubit_position)

    return stabilizer_state


def y_gate(stabilizer_state, qubit_position):
    """
    Pauli Y (=PXZP) applied on a single qubit given its position, in a stabilizer state.

    :param stabilizer_state: a StabilizerState object.
    :type stabilizer_state: StabilizerState
    :param qubit_position: index of the qubit that the gate acts on
    :type qubit_position: int
    :return: the resulting state after gate action
    :rtype: StabilizerState
    """
    n_qubits = stabilizer_state.n_qubits  # number of qubits
    assert qubit_position < n_qubits
    stabilizer_state = phase_gate(stabilizer_state, qubit_position)
    stabilizer_state = z_gate(stabilizer_state, qubit_position)
    stabilizer_state = x_gate(stabilizer_state, qubit_position)
    stabilizer_state = phase_gate(stabilizer_state, qubit_position)

    return stabilizer_state


def control_x_gate(stabilizer_state, ctrl_qubit, target_qubit):
    """
    Controlled X gate on control and target qubits given their position, in a stabilizer state.

    :param stabilizer_state: a StabilizerState object.
    :type stabilizer_state: StabilizerState
    :param ctrl_qubit: index of the control qubit
    :type ctrl_qubit: int
    :param target_qubit: index of the target qubit that the gate acts on
    :type target_qubit: int
    :return: the resulting state after gate action
    :rtype: StabilizerState
    """
    return cnot_gate(stabilizer_state, ctrl_qubit, target_qubit)


def control_z_gate(stabilizer_state, ctrl_qubit, target_qubit):
    """
    Controlled Z gate on control and target qubits given their position, in a stabilizer state.

    :param stabilizer_state: a StabilizerState object.
    :type stabilizer_state: StabilizerState
    :param ctrl_qubit: index of the control qubit
    :type ctrl_qubit: int
    :param target_qubit: index of the target qubit that the gate acts on
    :type target_qubit: int
    :return: the resulting state after gate action
    :rtype: StabilizerState
    """
    stabilizer_state = hadamard_gate(stabilizer_state, target_qubit)
    stabilizer_state = cnot_gate(stabilizer_state, ctrl_qubit, target_qubit)
    stabilizer_state = hadamard_gate(stabilizer_state, target_qubit)
    return stabilizer_state


def control_y_gate(stabilizer_state, ctrl_qubit, target_qubit):
    """
    Controlled Y gate on control and target qubits given their position, in a stabilizer state.

    :param stabilizer_state: a StabilizerState object.
    :type stabilizer_state: StabilizerState
    :param ctrl_qubit: index of the control qubit
    :type ctrl_qubit: int
    :param target_qubit: index of the target qubit that the gate acts on
    :type target_qubit: int
    :return: the resulting state after gate action
    :rtype: StabilizerState
    """
    stabilizer_state = phase_gate(stabilizer_state, target_qubit)
    stabilizer_state = z_gate(stabilizer_state, target_qubit)
    stabilizer_state = cnot_gate(stabilizer_state, ctrl_qubit, target_qubit)
    stabilizer_state = phase_gate(stabilizer_state, target_qubit)
    return stabilizer_state


def projector_z0(stabilizer_state, qubit_position):
    success = True
    n_qubits = stabilizer_state.n_qubits  # number of qubits
    assert qubit_position < n_qubits
    stabilizer_state, outcome, probabilistic = z_measurement_gate(
        stabilizer_state, qubit_position, seed=0
    )
    if probabilistic:
        if outcome == 0:
            pass
        else:
            stabilizer_state.tableau[probabilistic, -1] = 0
    else:
        if outcome == 0:
            pass
        else:
            stabilizer_state = 0
            success = False
            # TODO: see how impossible projection is handled in the density matrix formalism
    return stabilizer_state, success


def projector_z1(stabilizer_state, qubit_position):
    success = True
    n_qubits = stabilizer_state.n_qubits  # number of qubits
    assert qubit_position < n_qubits
    stabilizer_state, outcome, probabilistic = z_measurement_gate(
        stabilizer_state, qubit_position, seed=0
    )
    if probabilistic:
        if outcome == 1:
            pass
        else:
            stabilizer_state.tableau[probabilistic, -1] = 1
    else:
        if outcome == 1:
            pass
        else:
            stabilizer_state = 0
            success = False
            # TODO: see how impossible projection is handled in the density matrix formalism
    return stabilizer_state, success


def reset_qubit(stabilizer_state, qubit_position, intended_state):
    # reset qubit to computational basis states
    # NOTE: Only works after a measurement gate on the same qubit or for isolated qubits.
    n_qubits = stabilizer_state.n_qubits  # number of qubits
    assert qubit_position < n_qubits
    assert intended_state == 0 or intended_state == 1
    stabilizer_state, outcome, probabilistic = z_measurement_gate(
        stabilizer_state, qubit_position, seed=0
    )
    if probabilistic:
        if outcome == intended_state:
            pass
        else:
            stabilizer_state.tableau[probabilistic, -1] = intended_state
    else:
        if outcome == intended_state:
            pass
        else:
            the_generator = np.zeros(2 * n_qubits)
            the_generator[n_qubits + qubit_position] = 1

            generators = stabilizer_state.tableau[
                n_qubits : 2 * n_qubits, 0 : 2 * n_qubits
            ]
            for i in range(n_qubits):
                if generators[i] == the_generator:
                    assert stabilizer_state.tableau[i, qubit_position] == 1, (
                        "unexpected destabilizer element. The "
                        "reset gate in probably not used after a"
                        " valid measurement on the same qubit "
                    )
                    stabilizer_state.tableau[i, -1] = (
                        1 ^ stabilizer_state.tableau[i, -1]
                    )
    return stabilizer_state


def set_qubit(qubit, intended_state):
    """probably no possible to implement"""
    pass


def add_qubit(stabilizer_state):
    """
    add one isolated qubit in 0 state of the computational basis to the current state.

    :return: the updated stabilizer state
    """
    n_qubits = stabilizer_state.n_qubits  # number of qubits
    n_new = 1 + n_qubits
    if n_qubits == 0:
        return create_n_product_state(1)
    tableau = stabilizer_state.tableau
    new_tableau = create_n_product_state(1 + n_qubits).tableau
    # x destabilizer part
    new_tableau[0:n_qubits, 0:n_qubits] = tableau[0:n_qubits, 0:n_qubits]
    # z destabilizer part
    new_tableau[0:n_qubits, n_new : 2 * n_qubits + 1] = tableau[
        0:n_qubits, n_qubits : 2 * n_qubits
    ]
    # r destabilizer part
    new_tableau[0:n_qubits, -1] = tableau[0:n_qubits, -1]
    # x stabilizer part
    new_tableau[n_new : 2 * n_qubits + 1, 0:n_qubits] = tableau[
        n_qubits : 2 * n_qubits, 0:n_qubits
    ]
    # z stabilizer part
    new_tableau[n_new : 2 * n_qubits + 1, n_new : 2 * n_qubits + 1] = tableau[
        n_qubits : 2 * n_qubits, n_qubits : 2 * n_qubits
    ]
    # r stabilizer part
    new_tableau[n_new : 2 * n_qubits + 1, -1] = tableau[n_qubits : 2 * n_qubits, -1]

    stabilizer_state.n_qubits = n_new
    stabilizer_state.tableau = new_tableau
    return stabilizer_state


def insert_qubit(stabilizer_state, insert_after):
    """
    To be implemented. Similar to `add qubit` function.
    :param stabilizer_state:
    :param insert_after: the position index after which a new qubit is added.
    :return: updated state
    """


def remove_qubit(stabilizer_state, qubit_position):
    """
    Needs further investigation. Code below only works for isolated qubits.
    #TODO: Check if a qubit is isolated in general? Only isolated qubits in the Z basis states can be confirmed for now.
    The action of the function is measure and remove. If isolated, state should not change.
    only works correctly for isolated qubits! e.g. after measurement
    entangled qubits cannot be removed without affecting other parts of the state


    NEW method: discard any of the eligible rows from the stabilizer and destabilizer sets instead of finding THE one.
    """

    new_stabilizer_state, _, probabilistic = z_measurement_gate(
        stabilizer_state, qubit_position, seed=0
    )
    if probabilistic:
        row_position = probabilistic
    else:
        pass
    # number of qubits
    n_qubits = stabilizer_state.n_qubits
    assert qubit_position < n_qubits
    tableau = stabilizer_state.tableau
    tableau = np.delete(tableau, qubit_position, axis=1)
    tableau = np.delete(tableau, n_qubits + qubit_position, axis=1)
    row_positions = [i for i in range(len(tableau)) if not np.any(tableau[i])]
    assert len(row_positions) == 1
    row_position = row_positions[0]
    # order of removing rows matters here
    tableau = np.delete(tableau, n_qubits + row_position)
    tableau = np.delete(tableau, row_position)
    stabilizer_state.tableau = tableau

    return stabilizer_state


def trace_out():
    pass


def from_graph():
    pass


def from_stabilizer():
    pass


def swap_gate(stabilizer_state, qubit1, qubit2):
    pass


def is_pure():
    pass


def is_graph():
    pass


def is_stabilizer():
    pass


def create_n_product_state(n_qubits):
    """
    Creates a product state that consists :math:`n` tensor factors of the computational (Pauli Z) 0 state.
    """
    stabilizer_state = ss.Stabilizer()
    tableau = np.eye(2 * n_qubits)
    r_vector = np.zeros(2 * n_qubits).reshape(2 * n_qubits, 1)
    stabilizer_state.tableau = np.append(tableau, r_vector, axis=1)
    return stabilizer_state


def create_n_plus_state(n_qubits):
    """
    Creates a product state that consists :math:`n` tensor factors of the "plus state" (Pauli X's +1 eigenstate)
    """
    stabilizer_state = create_n_product_state(n_qubits)
    tableau = stabilizer_state.tableau
    tableau[:, [*range(2 * n_qubits)]] = tableau[
        :, [*range(n_qubits, 2 * n_qubits)] + [*range(0, n_qubits)]
    ]
    stabilizer_state.tableau = tableau
    return stabilizer_state


def tensor(list_of_states):
    pass


def partial_trace():
    pass


def fidelity(a, b):
    pass


def project_to_z0_and_remove(
    stabilizer_state, locations
):  # what if it cannot be projected to z0 ?! how is it
    # handled in density matrix formalism? ask
    pass


def measure_x(stabilizer_state, qubit_position, seed=0):
    """
    Returns the outcome 0 or 1 if one measures the given qubit in the X basis.
    NOTE: cannot update the stabilizer state after measurement. Stabilizer formalism can only handle Z-measurements.

    :param stabilizer_state: a StabilizerState object.
    :type stabilizer_state: StabilizerState
    :param qubit_position: index of the qubit that the gate acts on
    :type qubit_position: int
    :param seed: a seed for random outcome of the measurement
    :type seed: int
    :return: the classical outcome of measuring given qubit in the X basis.
    :rtype: int
    """
    stabilizer_state_new = hadamard_gate(stabilizer_state, qubit_position)
    _, outcome, _ = z_measurement_gate(stabilizer_state_new, qubit_position, seed=seed)
    return outcome


def measure_y(stabilizer_state, qubit_position, seed=0):
    """
    Returns the outcome 0 or 1 if one measures the given qubit in the Y basis.
    NOTE: cannot update the stabilizer state after measurement. Stabilizer formalism can only handle Z-measurements.

    :param stabilizer_state: a StabilizerState object.
    :type stabilizer_state: StabilizerState
    :param qubit_position: index of the qubit that the gate acts on
    :type qubit_position: int
    :param seed: a seed for random outcome of the measurement
    :type seed: int
    :return: the classical outcome of measuring given qubit in the X basis.
    :rtype: int
    """
    stabilizer_state_new = stabilizer_state
    # apply P dagger gate
    stabilizer_state_new = phase_dagger_gate(stabilizer_state_new, qubit_position)
    # apply H
    stabilizer_state_new = hadamard_gate(stabilizer_state_new, qubit_position)

    _, outcome, _ = z_measurement_gate(stabilizer_state_new, qubit_position, seed=seed)
    return outcome


def measure_z(stabilizer_state, qubit_position, seed=0):
    """
    Returns the outcome 0 or 1 if one measures the given qubit in the X basis.
    NOTE: Does not update the stabilizer state after measurement.

    :param stabilizer_state: a StabilizerState object.
    :type stabilizer_state: StabilizerState
    :param qubit_position: index of the qubit that the gate acts on
    :type qubit_position: int
    :param seed: a seed for random outcome of the measurement
    :type seed: int
    :return: the classical outcome of measuring given qubit in the X basis.
    :rtype: int
    """
    stabilizer_state_new = stabilizer_state
    _, outcome, _ = z_measurement_gate(stabilizer_state_new, qubit_position, seed=seed)
    return outcome


#######################################################################################


def hadamard_transform(x_matrix, z_matrix, positions):
    """
    Apply a Hadamard gate on each qubit specified by positions. This action is equivalent to a
    column swap between X matrix and Z matrix for the corresponding qubits.
    (not a stabilizer backend quantum gate, just a helper function)

    :param x_matrix: X part of the symplectic representation
    :type x_matrix: numpy.ndarray
    :param z_matrix: Z part of the symplectic representation
    :type z_matrix: numpy.ndarray
    :param positions: positions of qubits where the Hadamard gates are applied
    :type positions: list[int]
    :rtype: numpy.ndarray, numpy.ndarray
    :return: the resulting X matrix and Z matrix
    """
    temp1 = list(z_matrix[:, positions])
    temp2 = list(x_matrix[:, positions])
    z_matrix[:, positions] = temp2
    x_matrix[:, positions] = temp1
    return x_matrix, z_matrix


def row_reduction(x_matrix, z_matrix):
    """
    Turns the x_matrix into a row reduced echelon form. Applies same row operations on z_matrix.

    :param x_matrix: binary matrix for representing Pauli X part of the symplectic binary
        representation of the stabilizer generators.
    :type x_matrix: numpy.ndarray
    :param z_matrix:binary matrix for representing Pauli Z part of the
        symplectic binary representation of the stabilizer generators
    :type z_matrix: numpy.ndarray
    :return: a tuple of the transformed x_matrix and z_matrix and the index of the last non-zero row of the new x_matrix
    :rtype: tuple(numpy.ndarray, numpy.ndarray, int)
    """

    pivot = [0, 0]
    old_pivot = [1, 1]

    while pivot[1] != old_pivot[1]:
        # all row reduction operations will at least change the column of the pivot by 1 (not true for its row! due
        # to last column pivot)
        old_pivot = pivot
        x_matrix, z_matrix, pivot = _row_red_one_step(x_matrix, z_matrix, pivot)
    return x_matrix, z_matrix, pivot[0]


def _row_red_one_step(x_matrix, z_matrix, pivot):
    """
    A helper function to apply one step of the row reduction algorithm, only on the pivot provided here.
    It is used in the main row reduction function.

    :param x_matrix: binary matrix for representing Pauli X part of the symplectic binary
        representation of the stabilizer generators
    :type x_matrix: numpy.ndarray
    :param z_matrix:binary matrix for representing Pauli Z part of the
        symplectic binary representation of the stabilizer generators
    :type z_matrix: numpy.ndarray
    :param pivot: a location in the input matrix
    :type pivot: list[int]
    :return: a tuple of the transformed x_matrix and z_matrix and the new pivot location
    :rtype: tuple(numpy.ndarray, numpy.ndarray, list[int])
    """
    n_row, n_column = np.shape(x_matrix)
    if pivot[1] == (n_column - 1):
        the_ones = []
        for i in range(pivot[0], n_row):
            if x_matrix[i, pivot[1]] == 1:
                the_ones.append(i)
        if not the_ones:
            # empty under (and including) pivot element on last column
            pivot[0] = pivot[0] - 1
        else:
            x_matrix = row_swap(x_matrix, the_ones[0], pivot[0])
            z_matrix = row_swap(z_matrix, the_ones[0], pivot[0])
            the_ones.remove(the_ones[0])
            for j in the_ones:
                x_matrix = add_rows(x_matrix, pivot[0], j)
                z_matrix = add_rows(z_matrix, pivot[0], j)
        return x_matrix, z_matrix, pivot
    elif pivot[0] == (n_row - 1):
        if x_matrix[pivot[0], pivot[1]] == 1:
            return x_matrix, z_matrix, pivot
        else:
            pivot = [pivot[0], pivot[1] + 1]
            return x_matrix, z_matrix, pivot

    else:
        # list of rows with value 1 under the pivot element
        the_ones = []
        for i in range(pivot[0], n_row):
            if x_matrix[i, pivot[1]] == 1:
                the_ones.append(i)
        # check if the column below is empty to skip it
        if not the_ones:
            pivot = [pivot[0], pivot[1] + 1]
            return x_matrix, z_matrix, pivot
        else:
            x_matrix = row_swap(x_matrix, the_ones[0], pivot[0])
            z_matrix = row_swap(z_matrix, the_ones[0], pivot[0])
            the_ones.remove(the_ones[0])
            for j in the_ones:
                x_matrix = add_rows(x_matrix, pivot[0], j)
                z_matrix = add_rows(z_matrix, pivot[0], j)
            pivot = [pivot[0] + 1, pivot[1] + 1]
            return x_matrix, z_matrix, pivot


def get_stabilizer_element_by_string(generator):
    """
    Return the corresponding tensor of Pauli matrices for the stabilizer generator specified by the input string

    :param generator: a string for one stabilizer element
    :type generator: str
    :return: a matrix representation of the stabilizer element
    :rtype: numpy.ndarray
    """
    stabilizer_elem = 1
    for pauli in generator:
        if pauli.lower() == "x":
            stabilizer_elem = dmf.tensor([stabilizer_elem, dmf.sigmax()])
        elif pauli.lower() == "y":
            stabilizer_elem = dmf.tensor([stabilizer_elem, dmf.sigmay()])
        elif pauli.lower() == "z":
            stabilizer_elem = dmf.tensor([stabilizer_elem, dmf.sigmaz()])
        else:
            stabilizer_elem = dmf.tensor([stabilizer_elem, np.eye(2)])

    return stabilizer_elem


def g_function(x_1, z_1, x_2, z_2):
    """
    A helper function to use in rowsum function. Takes 4 bits (2 pauli matrices in binary representation) as input and
    returns the phase factor needed when the two Pauli matrices are multiplied: Pauli_1 * Pauli_2

    Refer to section III of arXiv:quant-ph/0406196v5

    :param x_1: the x bit of the first Pauli operator
    :type x_1: int
    :param z_1: the z bit of the first Pauli operator
    :type z_1: int
    :param x_2: the x bit of the second Pauli operator
    :type x_2: int
    :param z_2: the z bit of the second Pauli operator
    :type z_2: int
    :return: the exponent k in the phase factor: i^k where "i" is the unit imaginary number.
    :rtype: int
    """
    if not (x_1 or z_1):  # both equal to zero
        return 0
    if x_1 and z_1:
        return (z_2 - x_2) % 4
    if x_1 == 1 and z_1 == 0:
        return (z_2 * (2 * x_2 - 1)) % 4
    if x_1 == 0 and z_1 == 1:
        return (x_2 * (1 - 2 * z_2)) % 4


def row_sum(x_matrix, z_matrix, r_vector, row_to_add, target_row):
    """
    Takes the full stabilizer tableau as input and sets the stabilizer generator in the target_row equal to
    (row_to_add + target_row) while keeping track  of the phase factor by updating the r_vector.
    This is based on the section III of the article arXiv:quant-ph/0406196v5

    :param x_matrix: x matrix in the symplectic representation of the stabilizers
    :type x_matrix: np.ndarray
    :param z_matrix: z matrix in the symplectic representation of the stabilizers
    :type z_matrix: np.ndarray
    :param r_vector: the vector of phase factors.
    :type r_vector: np.ndarray
    :param row_to_add: the stabilizer to multiply the target stabilizer with
    :type row_to_add: int
    :param target_row: the stabilizer to be multiplied by the "to_add" stabilizer
    :type target_row: int
    :return: updated stabilizer tableau
    :rtype: numpy.ndarray, numpy.ndarray, numpy.ndarray
    """
    number_of_qubits = np.shape(x_matrix)[0]
    # determining the phase factor
    g_sum = 0
    for j in range(number_of_qubits):
        g_sum = g_sum + g_function(
            x_matrix[row_to_add, j],
            z_matrix[row_to_add, j],
            x_matrix[target_row, j],
            z_matrix[target_row, j],
        )
    if (2 * r_vector[target_row, 0] + 2 * r_vector[row_to_add, 0] + g_sum) % 4 == 0:
        r_vector[target_row, 0] = 0
    elif (2 * r_vector[target_row, 0] + 2 * r_vector[row_to_add, 0] + g_sum) % 4 == 2:
        r_vector[target_row, 0] = 1
    else:
        raise Exception("input cannot be valid, due to unexpected outcome")

    # calculating the resulting new matrices after adding row i to h.
    x_matrix = add_rows(x_matrix, row_to_add, target_row)
    z_matrix = add_rows(z_matrix, row_to_add, target_row)

    return x_matrix, z_matrix, r_vector


def row_swap_full(x_matrix, z_matrix, r_vector, first_row, second_row):
    """
    swaps the rows of the full stabilizer tableau (including the phase factor vector)

    :param x_matrix: x matrix in the symplectic representation of the stabilizers
    :type x_matrix: np.ndarray
    :param z_matrix: z matrix in the symplectic representation of the stabilizers
    :type z_matrix: np.ndarray
    :param r_vector: the vector of phase factors.
    :type r_vector: np.ndarray
    :param first_row: one of the rows to be swapped
    :type first_row: int
    :param second_row: the other row to be swapped
    :type second_row: int
    :return: updated stabilizer tableau
    :rtype: numpy.ndarray, numpy.ndarray, numpy.ndarray
    """
    x_matrix = row_swap(x_matrix, first_row, second_row)
    z_matrix = row_swap(z_matrix, first_row, second_row)
    r_vector = row_swap(r_vector, first_row, second_row)
    return x_matrix, z_matrix, r_vector


def pauli_type_finder(x_matrix, z_matrix, pivot):
    """
    A function that counts the types and the number of the Pauli operators that are present on and below an element
    (the pivot) in the stabilizer tableau.

    :param x_matrix: x matrix in the symplectic representation of the stabilizers
    :type x_matrix: np.ndarray
    :param z_matrix: z matrix in the symplectic representation of the stabilizers
    :type z_matrix: np.ndarray
    :param pivot: the location of the pivot element [i,j] on the i-th row and j-th column of the stabilizer tableau.
    :type pivot: list
    :return: three lists each containing the positions (row indices) of the Pauli X, Y, and Z operators below the pivot,
    for example, if the first list is [3, 4] it means there are Pauli X operators in rows 3 and 4 in the pivot column.
    :rtype: list, list, list
    """
    n_qubits = np.shape(x_matrix)[0]
    # list of the rows (generators) with a pauli X operator in the pivot column
    pauli_x_list = []
    # list of the rows (generators) with a pauli Y operator in the pivot column
    pauli_y_list = []
    # list of the rows (generators) with a pauli Z operator in the pivot column
    pauli_z_list = []

    for row_i in range(pivot[0], n_qubits):
        if x_matrix[row_i, pivot[1]] == 1 and z_matrix[row_i, pivot[1]] == 0:
            pauli_x_list.append(row_i)
        if x_matrix[row_i, pivot[1]] == 1 and z_matrix[row_i, pivot[1]] == 1:
            pauli_y_list.append(row_i)
        if x_matrix[row_i, pivot[1]] == 0 and z_matrix[row_i, pivot[1]] == 1:
            pauli_z_list.append(row_i)

    return pauli_x_list, pauli_y_list, pauli_z_list


def _process_one_pauli(x_matrix, z_matrix, r_vector, pivot, pauli_list):
    """
    Helper function to process one Pauli list

    :param x_matrix:
    :param z_matrix:
    :param r_vector:
    :param pivot:
    :param pauli_list:
    :return:
    """
    x_matrix, z_matrix, r_vector = row_swap_full(
        x_matrix, z_matrix, r_vector, pivot[0], pauli_list[0]
    )

    # remove the first element of the list
    pauli_list = pauli_list[1:]

    for row_i in pauli_list:
        # multiplying rows with similar pauli to eliminate them
        x_matrix, z_matrix, r_vector = row_sum(
            x_matrix, z_matrix, r_vector, pivot[0], row_i
        )

    pivot = [pivot[0] + 1, pivot[1] + 1]
    return x_matrix, z_matrix, r_vector, pivot


def _process_two_pauli(
    x_matrix,
    z_matrix,
    r_vector,
    pivot,
    pauli_list_dict,
    first_list_symbol,
    second_list_symbol,
):
    """
    Helper function to process two Pauli lists

    :param x_matrix:
    :param z_matrix:
    :param r_vector:
    :param pivot:
    :param pauli_list_dict:
    :param first_list_symbol:
    :param second_list_symbol:
    :return:
    """
    # swap the pivot and its next row with them

    x_matrix, z_matrix, r_vector = row_swap_full(
        x_matrix, z_matrix, r_vector, pivot[0], pauli_list_dict[first_list_symbol][0]
    )
    # update pauli lists
    pauli_x_list, pauli_y_list, pauli_z_list = pauli_type_finder(
        x_matrix, z_matrix, pivot
    )

    pauli_list_dict["x"] = pauli_x_list
    pauli_list_dict["y"] = pauli_y_list
    pauli_list_dict["z"] = pauli_z_list
    x_matrix, z_matrix, r_vector = row_swap_full(
        x_matrix,
        z_matrix,
        r_vector,
        pivot[0] + 1,
        pauli_list_dict[second_list_symbol][0],
    )
    # update pauli lists
    pauli_x_list, pauli_y_list, pauli_z_list = pauli_type_finder(
        x_matrix, z_matrix, pivot
    )
    pauli_list_dict["x"] = pauli_x_list
    pauli_list_dict["y"] = pauli_y_list
    pauli_list_dict["z"] = pauli_z_list
    assert (
        pauli_list_dict[first_list_symbol][0] == pivot[0]
        and pauli_list_dict[second_list_symbol][0] == pivot[0] + 1
    ), "row operations failed"

    # remove the first element of the list
    pauli_list_dict[first_list_symbol] = pauli_list_dict[first_list_symbol][1:]
    pauli_list_dict[second_list_symbol] = pauli_list_dict[second_list_symbol][1:]

    for row_i in pauli_list_dict[first_list_symbol]:
        # multiplying rows with similar pauli to eliminate them
        x_matrix, z_matrix, r_vector = row_sum(
            x_matrix, z_matrix, r_vector, pivot[0], row_i
        )

    for row_j in pauli_list_dict[second_list_symbol]:
        # multiplying rows with similar pauli to eliminate them
        x_matrix, z_matrix, r_vector = row_sum(
            x_matrix, z_matrix, r_vector, pivot[0] + 1, row_j
        )

    pivot = [pivot[0] + 2, pivot[1] + 1]
    return x_matrix, z_matrix, r_vector, pivot


def one_step_rref(x_matrix, z_matrix, r_vector, pivot):
    """
    ROW-REDUCED ECHELON FORM algorithm that takes the pivot element location and stabilizer tableau,
    and converts the elements below the pivot to the standard row echelon form.
    This is one of the steps of the full row reduced echelon form algorithm.

    :param x_matrix: x matrix in the symplectic representation of the stabilizers
    :type x_matrix: np.ndarray
    :param z_matrix: z matrix in the symplectic representation of the stabilizers
    :type z_matrix: np.ndarray
    :param r_vector: the vector of phase factors.
    :type r_vector: np.ndarray
    :param pivot: the location of the pivot element [i,j] on the i-th row and j-th column of the stabilizer tableau.
    :type pivot: list
    :return: updated stabilizer tableau and updated pivot
    :rtype: numpy.ndarray, numpy.ndarray, numpy.ndarray, list
    """

    # pauli_x_list  = list of the rows (generators) with a pauli X operator in the pivot column
    # pauli_y_list  = list of the rows (generators) with a pauli Y operator in the pivot column
    # pauli_z_list  = list of the rows (generators) with a pauli Z operator in the pivot column
    pauli_x_list, pauli_y_list, pauli_z_list = pauli_type_finder(
        x_matrix, z_matrix, pivot
    )
    pauli_list_dict = {"x": pauli_x_list, "y": pauli_y_list, "z": pauli_z_list}
    # case of no pauli operator!
    if not (pauli_x_list or pauli_y_list or pauli_z_list):
        pivot = [pivot[0], pivot[1] + 1]
        return x_matrix, z_matrix, r_vector, pivot

    # case of only 1 kind of pauli
    elif pauli_x_list and (not pauli_y_list) and (not pauli_z_list):  # only X
        return _process_one_pauli(x_matrix, z_matrix, r_vector, pivot, pauli_x_list)

    elif pauli_y_list and (not pauli_x_list) and (not pauli_z_list):  # only Y
        return _process_one_pauli(x_matrix, z_matrix, r_vector, pivot, pauli_y_list)

    elif pauli_z_list and (not pauli_x_list) and (not pauli_y_list):  # only Z
        return _process_one_pauli(x_matrix, z_matrix, r_vector, pivot, pauli_z_list)

    # case of two kinds of pauli
    elif not pauli_x_list:  # pauli y and z exist in the column below pivot
        return _process_two_pauli(
            x_matrix, z_matrix, r_vector, pivot, pauli_list_dict, "y", "z"
        )

    elif not pauli_y_list:  # pauli x and z exist in the column below pivot
        return _process_two_pauli(
            x_matrix, z_matrix, r_vector, pivot, pauli_list_dict, "x", "z"
        )

    elif not pauli_z_list:  # pauli x and y exist in the column below pivot
        return _process_two_pauli(
            x_matrix, z_matrix, r_vector, pivot, pauli_list_dict, "x", "y"
        )

    # case of all three kinds of paulis available in the column
    else:
        old_pivot = [0, 0]
        old_pivot[0] = pivot[0]
        old_pivot[1] = pivot[1]
        x_matrix, z_matrix, r_vector, _ = _process_two_pauli(
            x_matrix, z_matrix, r_vector, old_pivot, pauli_list_dict, "x", "z"
        )
        # update pauli lists
        pauli_x_list, pauli_y_list, pauli_z_list = pauli_type_finder(
            x_matrix, z_matrix, pivot
        )
        for row_k in pauli_y_list:
            # multiplying the pauli Y with pauli X to make it Z
            x_matrix, z_matrix, r_vector = row_sum(
                x_matrix, z_matrix, r_vector, pivot[0], row_k
            )
            # multiplying the now Z row with another Z to eliminate it
            x_matrix, z_matrix, r_vector = row_sum(
                x_matrix, z_matrix, r_vector, pivot[0] + 1, row_k
            )
        pivot = [pivot[0] + 2, pivot[1] + 1]
        return x_matrix, z_matrix, r_vector, pivot


def rref(x_matrix, z_matrix, r_vector):
    """
    Takes stabilizer tableau, and converts it to the standard row echelon form.

    :param x_matrix: x matrix in the symplectic representation of the stabilizers
    :type x_matrix: np.ndarray
    :param z_matrix: z matrix in the symplectic representation of the stabilizers
    :type z_matrix: np.ndarray
    :param r_vector: the vector of phase factors.
    :type r_vector: np.ndarray
    :return: stabilizer tableau in the row reduced echelon form
    :rtype: numpy.ndarray, numpy.ndarray, numpy.ndarray
    """
    # TODO: check the validity of input x and z matrices. Partially done by checking the rank by assertion below.
    pivot = [0, 0]
    n_qubits = np.shape(x_matrix)[0]
    while pivot[0] <= n_qubits - 1 and pivot[1] <= n_qubits - 1:
        x_matrix, z_matrix, r_vector, pivot = one_step_rref(
            x_matrix, z_matrix, r_vector, pivot
        )
    assert (
        pivot[0] >= n_qubits - 1
    ), "Invalid input. One of the stabilizers is identity on all qubits!"  # rank check
    return x_matrix, z_matrix, r_vector


def height_func_list(x_matrix, z_matrix):
    """
    Calculates the height_function for all qubit in the graph given the stabilizer tableau of a graph state with ordered
    nodes. Node ordering should correspond to the rows present in the adjacency matrix. (i-th node must be i-th row)

    :param x_matrix: x matrix in the symplectic representation of the stabilizers
    :type x_matrix: np.ndarray
    :param z_matrix: z matrix in the symplectic representation of the stabilizers
    :type z_matrix: np.ndarray
    :return: the height as a function of qubit positions in graph. This is related to the entanglement entropy with
    respect to the bi-partition of the state at the given position.
    :rtype: int
    """
    n_qubits = np.shape(x_matrix)[0]
    r_vector = np.zeros([n_qubits, 1])
    height_list = []

    x_matrix, z_matrix, r_vector = rref(x_matrix, z_matrix, r_vector)

    for qubit_position in range(n_qubits):
        left_most_nontrivial = []
        for row_i in range(n_qubits):
            for column_j in range(n_qubits):
                if not (
                    x_matrix[row_i, column_j] == 0 and z_matrix[row_i, column_j] == 0
                ):
                    left_most_nontrivial.append(column_j)
                    break
        assert len(left_most_nontrivial) == n_qubits, (
            "Invalid input. One of the stabilizers is identity on " "all qubits!"
        )
        n_non_trivial_generators = len(
            [x for x in left_most_nontrivial if x - qubit_position > 0]
        )
        height = n_qubits - (qubit_position + 1) - n_non_trivial_generators
        height_list.append(height)
    return height_list


def height_function(x_matrix, z_matrix, qubit_position):
    """
    Calculates the height_function for the desired qubit in the graph given the label (position) of the qubit/node.

    :param x_matrix: x matrix in the symplectic representation of the stabilizers
    :type x_matrix: np.ndarray
    :param z_matrix: z matrix in the symplectic representation of the stabilizers
    :type z_matrix: np.ndarray
    :param qubit_position: label or position of the qubit/node in the graph
    :type qubit_position: int
    :return: the height function at the given qubit. This is related to the entanglement entropy with respect to the
    bi-partition of the state at the given position.
    :rtype: int
    """

    height = height_func_list(x_matrix, z_matrix)[qubit_position]
    return height


def height_dict(x_matrix=None, z_matrix=None, graph=None):
    """
    Generates the height_function dictionary for all qubits, given the x and z matrices or the graph the state
    corresponds to.

    :param x_matrix: x matrix in the symplectic representation of the stabilizers
    :type x_matrix: np.ndarray
    :param z_matrix: z matrix in the symplectic representation of the stabilizers
    :type z_matrix: np.ndarray
    :param graph: the graph corresponding to the state
    :type graph: networkx.classes.graph.Graph
    :return: the value of the height function for all positions in a dictionary.
    :rtype: dict
    """
    if x_matrix is None or z_matrix is None:
        if isinstance(graph, nx.classes.graph.Graph):
            n_qubits = len(graph)
            node_list = list(graph.nodes()).sort()
            # nodelist is an essential kwarg in converting graph to adjacency matrix.
            z_matrix = nx.to_numpy_array(graph, nodelist=node_list)
            x_matrix = np.eye(n_qubits)
        elif graph:
            raise ValueError("graph should be a valid networkx graph object")
        else:
            raise ValueError(
                "Either a graph or both x AND z matrices must be provided."
            )

    n_qubits = np.shape(x_matrix)[0]
    positions = [-1] + [*range(n_qubits)]
    # the first element of qubit positions list is set to -1 for symmetric plotting of the height function.
    height_x = [0] + height_func_list(x_matrix, z_matrix)
    # the first element of height function is set to zero and corresponds to an imaginary qubit at position -1.

    h_dict = {positions[i]: height_x[i] for i in range(n_qubits + 1)}
    return h_dict


def height_max(x_matrix=None, z_matrix=None, graph=None):
    """
    Given the x and z matrices or the graph the state corresponds to. Returns the maximum of height function which is
    equal to the minimum number of emitter needed for deterministic generation of the state

    :param x_matrix: x matrix in the symplectic representation of the stabilizers
    :type x_matrix: np.ndarray
    :param z_matrix: z matrix in the symplectic representation of the stabilizers
    :type z_matrix: np.ndarray
    :param graph: the graph corresponding to the state
    :type graph: networkx.classes.graph.Graph
    :return: maximum of height function over all qubits.
    :rtype: int
    """
    h_dict = height_dict(x_matrix=x_matrix, z_matrix=z_matrix, graph=graph)
    h_max = h_dict[max(h_dict, key=h_dict.get)]
    return h_max


def height_plotter(h_dict):
    """
    Plots the height function.

    :param h_dict: the height function dict which is the output of the ``height_dict``.
    :type h_dict: dict
    :return: maximum of height function over all qubits.
    :rtype: int
    """
    h_max = h_dict[max(h_dict, key=h_dict.get)]
    positions = list(h_dict.keys())
    height_x = list(h_dict.values())
    number_of_qubits = len(positions) - 1
    fig1, ax1 = plt.subplots(1, 1, constrained_layout=True, sharey=True)
    ax1.plot(positions, height_x, marker="o", markerfacecolor="red", markersize=8)
    ax1.set_title("The height function")
    ax1.set_xlabel("qubit position")
    ax1.set_ylabel("Bipartite Entanglement")
    ax1.set(xlim=(-1, number_of_qubits - 1), ylim=(0, h_max + 1))
    ax1.set_yticks(range(0, h_max + 1))
    ax1.set_xticks(positions)
    plt.show()


"""
Functions related to stabilizer table verification
"""


def is_symplectic(tableau):
    """
    Check if a given tableau is symplectic self-orthogonal.

    :param tableau:
    :type tableau:
    :return:
    :rtype: bool
    """
    dim = int(tableau.shape[1] / 2)
    symplectic_p = np.block(
        [[np.zeros((dim, dim)), np.eye(dim)], [np.eye(dim), np.zeros((dim, dim))]]
    ).astype(int)

    return np.array_equal(binary_symplectic_product(tableau, tableau), symplectic_p)


def is_symplectic_self_orthogonal(tableau):
    """
    Check if a given tableau is symplectic self-orthogonal.

    :param tableau:
    :type tableau:
    :return:
    :rtype: bool
    """
    dim = int(tableau.shape[1] / 2)

    return np.array_equal(
        binary_symplectic_product(tableau, tableau), np.zeros((dim, dim))
    )


def binary_symplectic_product(matrix1, matrix2):
    """
    Compute the binary symplectic product of two matrices matrix1 (:math:`M_1`) and matrix2 (:math:`M_2`)

    The symplectic inner product between :math:`M_1` and :math:`M_2` is :math:`M_1 P M_2^T`,
    where :math:`P = \\begin{bmatrix} 0 & I \\\ I & 0 \\end{bmatrix}`.


    :param matrix1:
    :type matrix1:
    :param matrix2:
    :type matrix2:
    :return:
    :rtype:
    """
    assert matrix1.shape[0] == matrix2.shape[0]
    dim = matrix1.shape[0]
    symplectic_p = np.block(
        [[np.zeros((dim, dim)), np.eye(dim)], [np.eye(dim), np.zeros((dim, dim))]]
    ).astype(int)
    return ((matrix1 @ symplectic_p @ matrix2.T) % 2).astype(int)
