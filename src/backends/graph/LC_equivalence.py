import numpy as np
import networkx as nx
from itertools import combinations
from itertools import permutations
import matplotlib.pyplot as plt
from IPython.display import clear_output


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
    x, z = x_matrix, z_matrix
    pivot = [0, 0]
    old_pivot = [1, 1]

    while pivot[1] != old_pivot[1]:
        # all row reduction operations will at least change the column of the pivot by 1 (not true for its row! due
        # to last column pivot)
        old_pivot = pivot
        x, z, pivot = _row_red_one_step(x, z, pivot)
    return x, z, pivot[0]


def _row_red_one_step(x_matrix, z_matrix, pivot):  # one step of the algorithm, only on the pivot provided here
    """
    A helper function to apply one step of the row reduction algorithm. It is used in the main row reduction function.
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
    n, m = np.shape(x_matrix)
    if pivot[1] == (m - 1):
        the_ones = []
        for a in range(pivot[0], n):
            if x_matrix[a, pivot[1]] == 1:
                the_ones.append(a)
        if not the_ones:  # empty under (and including) pivot element on last column
            pivot[0] = pivot[0] - 1
        else:
            x_matrix = _row_swap(x_matrix, the_ones[0], pivot[0])
            z_matrix = _row_swap(z_matrix, the_ones[0], pivot[0])
            the_ones.remove(the_ones[0])
            for b in the_ones:
                x_matrix = _add_rows(x_matrix, pivot[0], b)
                z_matrix = _add_rows(z_matrix, pivot[0], b)
        return x_matrix, z_matrix, pivot
    elif pivot[0] == (n - 1):
        if x_matrix[pivot[0], pivot[1]] == 1:
            return x_matrix, z_matrix, pivot
        else:
            pivot = [pivot[0], pivot[1] + 1]
            return x_matrix, z_matrix, pivot

    else:
        # list of rows with value 1 under the pivot element
        the_ones = []
        for a in range(pivot[0], n):
            if x_matrix[a, pivot[1]] == 1:
                the_ones.append(a)
        # check if the column below is empty to skip it
        if not the_ones:
            pivot = [pivot[0], pivot[1] + 1]
            return x_matrix, z_matrix, pivot
        else:
            x_matrix = _row_swap(x_matrix, the_ones[0], pivot[0])
            z_matrix = _row_swap(z_matrix, the_ones[0], pivot[0])
            the_ones.remove(the_ones[0])
            for b in the_ones:
                x_matrix = _add_rows(x_matrix, pivot[0], b)
                z_matrix = _add_rows(z_matrix, pivot[0], b)
            pivot = [pivot[0] + 1, pivot[1] + 1]
            return x_matrix, z_matrix, pivot


# row reduction functions
def _row_swap(x_matrix, row_i, row_j):
    """
    Swaps two rows of a matrix
    :param x_matrix: a matrix
    :type x_matrix: numpy.ndarray
    :param row_i: the first row
    :type row_i: int
    :param row_j: the second row
    :type row_j: int
    :return: the matrix after swapping those two row
    :rtype: numpy.ndarray
    """
    x_matrix[[row_i, row_j]] = x_matrix[[row_j, row_i]]
    return x_matrix


def _add_rows(x_matrix, row_i, target_row):
    """
    Adds two rows together modulo 2 and put it in the row of the second input
    :param x_matrix: a binary matrix
    :type x_matrix: numpy.ndarray
    :param row_i: the index of the row to add
    :type row_i: int
    :param target_row: the index of the row where the result is put
    :type target_row: int
    :return: the matrix after adding two rows modulo 2 and putting in the row of the second input
    :rtype: numpy.ndarray
    """
    x_matrix[target_row] = (x_matrix[row_i] + x_matrix[target_row]) % 2
    return x_matrix


def solver(z_1, z_2, Mode='deterministic'):
    """
    Determines whether two graph states are local-Clifford equivalent or not, given the adjacency matrices of the two.
    :param z_1: The adjacency matrix of the first graph. This is equal to the binary matrix for representing Pauli Z
        part of the symplectic binary representation of the stabilizer generators
    :type z_1: numpy.ndarray
    :param z_2:The adjacency matrix of the second graph.
    :type z_2: numpy.ndarray
    :param Mode: the mode of the solver is chosen. It can be either 'deterministic' (default) or 'random'.
    :type Mode: str
    :return: If a solution is found, returns an array of single-qubit Clifford 2*2 matrices in the symplectic formalism.
        If not, graphs are not LC equivalent and returns None.
    :rtype: numpy.ndarray or None
    """
    # gets two adjacency matrices as input and returns a np.array containing n (2*2 array)s = clifford operations on
    # each qubit
    n = np.shape(z_1)[0]
    # get the coefficient matrix for the system of linear equations
    coeff_matrix = _coeff_maker(z_1, z_2)

    # check for rank to see how many independent equations are there = rank of the matrix
    rank = np.linalg.matrix_rank(coeff_matrix)
    if rank >= 4 * n:
        print(f'rank = {rank} >= 4n = {4 * n} Two graphs/states are not LC equivalent for sure')
        return

    reduced_coeff_matrix, b, c = row_reduction(coeff_matrix, coeff_matrix * 0)  # row reduction applied

    rank = c + 1
    reduced_coeff_matrix = np.array(
        [i for i in reduced_coeff_matrix if i.any()])  # update the matrix to remove zero rows
    assert (np.shape(reduced_coeff_matrix)[0] == rank), "remaining rows are less than the rank!"
    rank = np.shape(reduced_coeff_matrix)[0]

    col_list = _col_finder(reduced_coeff_matrix)  # finding linear dependent columns!
    length = len(col_list)
    assert (length == 4 * n - rank), "column list is not correct"

    # check for random solution 1000 times
    if Mode == 'random':
        rand_s = _random_checker(reduced_coeff_matrix, col_list, try_count=1000)
        if isinstance(rand_s, type(np.array([0]))):
            return rand_s.reshape(n, 2, 2)  # random result
        else:
            return

    elif Mode == 'deterministic':

        basis = _solution_basis_finder(reduced_coeff_matrix, col_list)
        sub_set = list(combinations(basis, 2))
        solution_set = []
        for x in sub_set:
            a = x[0] + x[1]
            solution_set.append(a % 2)
        for y in solution_set:
            if _verifier(y):
                print("solution found! \n")
                return y.reshape(n, 2, 2)  # convert the solution (=y) to an array of n * (2*2) matrices
        print("states are NOT LC equivalent")
        return
    else:
        print("the Mode should be either 'random' or 'deterministic' (default)")
        return


def _solution_basis_finder(reduced_coeff_matrix, col_list):
    """
    Finds the basis for all acceptable solutions in the system of linear equations.
    :param reduced_coeff_matrix: an echelon form matrix of the coefficients in the system of linear equations.
    :type reduced_coeff_matrix: numpy.ndarray
    :param col_list: list of the columns' indices in the coefficient matrix that are linearly dependent.
    :type col_list: list[int]
    :return: an array containing the elements of the total Clifford operation needed to convert one graph to another.
    :rtype: numpy.ndarray
    """

    length = len(col_list)
    bbb = []
    for c in col_list:
        b = reduced_coeff_matrix[:, c]
        b = b.reshape((np.shape(reduced_coeff_matrix)[1] - length, 1))
        bbb.append(b)
    b_matrix = np.array(bbb)  # contains all vectors b as its columns

    a_matrix = np.delete(reduced_coeff_matrix, col_list, axis=1)  # removing linear dependent columns!

    x_array = ((np.linalg.inv(a_matrix)) % 2 @ b_matrix) % 2
    x_array = x_array.astype(int)
    basis_list = np.eye(length).astype(int)
    llist = []
    for counter in range(length):
        llist.append(list((x_array[counter][:, 0])))
        for i in range(length):
            llist[counter].insert(col_list[i], basis_list[counter, i])
    v_array = np.array(llist)
    v_array = v_array.reshape((length, np.shape(reduced_coeff_matrix)[1], 1))

    print(f'solution basis is full rank = {np.shape(v_array)[0]} :',
          np.linalg.matrix_rank(v_array[:, :, 0]) == np.shape(v_array)[0])
    # check also that it give zero array as result:
    assert not (((reduced_coeff_matrix @ v_array) % 2).any()), "solution basis is wrong!"

    return v_array.astype(int)


def _random_checker(reduced_coeff_matrix, col_list, try_count=10000):
    """
    Randomly searches for solutions to the system of linear equations for finding the Clifford operation matrix elements
    :param reduced_coeff_matrix: an echelon form matrix of the coefficients in the system of linear equations.
    :type reduced_coeff_matrix: numpy.ndarray
    :param col_list: list of the columns' indices in the coefficient matrix that are linearly dependent.
    :type col_list: list[int]
    :return: if successful, an array containing the elements of the total Clifford operation needed to convert one graph
        to another. If not, returns None.
    :rtype: numpy.ndarray or None
    """
    n = int(np.shape(reduced_coeff_matrix)[1] / 4)
    length = len(col_list)

    # rand_var_vec = a random choice of the variables' vector for the n-rank parameters that the equations cannot
    # handle!

    for j in range(try_count):
        rand_var_vec = np.zeros((4 * n, 1))
        for i in range(length):
            rand_var_vec[col_list[i]] = np.random.randint(2, size=(1, 1))[0, 0]
        solution = _vec_solution_finder(reduced_coeff_matrix, col_list, rand_var_vec)
        if _verifier(solution):
            print("Random solution found!")
            return solution
    print("Random search unsuccessful")
    return


def _vec_solution_finder(reduced_coeff_matrix, col_list, var_vec):
    """
    Based on an input vector for variables that can take any value  in the solution of the system of equations (due to
        being under defined), finds one of the vectors in the basis for all possible solutions.
    :param reduced_coeff_matrix: an echelon form matrix of the coefficients in the system of linear equations.
    :type reduced_coeff_matrix: numpy.ndarray
    :param col_list: list of the columns' indices in the coefficient matrix that are linearly dependent.
    :type col_list: list[int]
    :param var_vec: a vector of arbitrary variable values.
    :type var_vec: numpy.ndarray
    :return: an array that is one of the vectors in the solution basis for the system of linear equations.
    :rtype: numpy.ndarray
    """
    # find the matrix equation Ax=b A is the square(rank x rank) matrix out of the reduced_coeff_matrix x is the
    # vector of length=rank to be found by A^(-1)*b.
    # b is the vector obtained from randomly choosing the extra
    # unknowns of vector rand_vec: b= (-1)* reduced_coeff_matrix * var_vec

    n = int(np.shape(reduced_coeff_matrix)[1] / 4)
    a = np.delete(reduced_coeff_matrix, col_list, axis=1)  # removing linear dependent columns!
    b = (reduced_coeff_matrix @ var_vec) % 2
    x = ((np.linalg.inv(a)) % 2 @ b) % 2
    # the full var_vec is now the x vector inserted to the var_vec vector to make all 4*n elements
    counter = 0
    for i in range(4 * n):
        if i not in col_list:
            var_vec[i] = x[i - counter][0]
        else:
            counter = counter + 1

    return var_vec.astype(int)


def _verifier(vector):
    """
    Verifies if the given matrix elements for a Clifford operation is valid.(if the Clifford matrix is invertible)
    :param vector: an array containing the matrix elements of the total Clifford operation needed to convert one graph
        to another.
    :type vector: numpy.ndarray
    :return: True if the input is valid and False if it is not.
    :rtype: bool
    """
    # reshapes a 4*n vector into an array of 2*2 matrices which are the a_i, b_i, c_i, d_i  elements in Q = Clifford
    n = int(np.shape(vector)[0] / 4)
    v = vector.reshape(n, 2, 2)
    checklist = []
    for i in range(n):
        a = (v[i][0, 0] * v[i][1, 1]) + (v[i][0, 1] * v[i][1, 0])  # XOR
        checklist.append(int(a % 2))
    return all(checklist)


def _coeff_maker(z1_matrix, z2_matrix):
    """
    Forms the coefficient matrix for the system of linear equations to find the matrix elements of the Clifford
        operation needed to convert the initial graph to the target graph, given the adjacency matrices of the two.
    :param z1_matrix: The adjacency matrix of the first graph. This is equal to the binary matrix for representing
        Pauli Z part of the symplectic binary representation of the stabilizer generators
    :type z1_matrix: numpy.ndarray
    :param z2_matrix:The adjacency matrix of the second graph.
    :type z2_matrix: numpy.ndarray
    :return: the coefficient matrix for the system of linear equations for the Clifford operation matrix elements
    :rtype: numpy.ndarray
    """
    # z1 and z2 are initial and target adjacency matrices.
    # Returns the coefficient matrix for system of n**2 linear equations.
    n = np.shape(z1_matrix)[0]
    assert (np.shape(z1_matrix)[0] == np.shape(z2_matrix)[0]), "graphs must be of same size"

    coeff_matrix = np.zeros((n ** 2, 4 * n)).astype(int)
    for j in range(n):
        for k in range(n):
            for m in range(n):
                row = (n * j + k)
                # a_m
                if m == k:
                    coeff_matrix[row, 4 * m + 0] = z1_matrix[j, k]
                    # b_m
                if m == k and j == k:
                    coeff_matrix[row, 4 * m + 1] = 1
                # c_m
                coeff_matrix[row, 4 * m + 2] = z1_matrix[m, j] * z2_matrix[m, k]
                # d_m
                if m == j:
                    coeff_matrix[row, 4 * m + 3] = z2_matrix[j, k]
    return coeff_matrix % 2


def _col_finder(x):
    """
    Finds linearly dependent columns in a row reduced matrix.
    :param x: this is the row reduced coefficient matrix
    :type x: numpy.ndarray
    :return: a list of the indices of the columns which are not linearly independent
    :rtype: numpy.ndarray
    """
    the_list = []
    pivot = [0, 0]
    m, k = np.shape(x)
    for i in range(k - 1):
        # print(pivot)
        if x[pivot[0], pivot[1]] == 1:

            if pivot[0] == (m - 1):
                pivot = [pivot[0], pivot[1] + 1]
                the_list.extend([*range(pivot[1], k)])
                break
            else:
                pivot = [pivot[0] + 1, pivot[1] + 1]

        elif x[pivot[0], pivot[1]] == 0:
            the_list.append(pivot[1])
            pivot = [pivot[0], pivot[1] + 1]
        else:
            raise ValueError('elements of matrix should be 0 or 1 only')
    return the_list


def local_clifford_ops(solution):
    """
    Finds a list of operators needed to be applied on each qubit of the first graph to transform in to the second,
        given the Clifford transformation matrix, which is the output of the solver function.
    :param solution: an array of single-qubit Clifford 2*2 matrices in the symplectic formalism
    :type solution: numpy.ndarray
    :return: a list of the names of the operations that need to be applied on each qubit in the correct order.
    :rtype: list[str]
    """
    # The order of the operations is the same as the qubits' labels in the graphs

    # allowed operations on single qubits
    identity = np.array([[1, 0], [0, 1]])
    hadamard = np.array([[0, 1], [1, 0]])
    phase = np.array([[1, 1], [0, 1]])
    ph = np.array([[1, 1], [1, 0]])
    hp_dagger = np.array([[0, 1], [1, 1]])
    php = np.array([[1, 0], [1, 1]])

    ops_list = [identity, hadamard, phase, ph, hp_dagger, php]
    ops_list_str = ['I', 'H', 'P', 'PH', 'H P_dag', 'PHP']
    ops_dict = zip(list(range(len(ops_list))), ops_list_str)
    ops_dict = dict(ops_dict)
    ops_names = []
    for i in solution:
        # a = [ops_list.index(j) for j in ops_list if (i==j).all()] # a size = 1 list, contains the index of operation
        for j in range(len(ops_list)):
            if np.array_equal(i, ops_list[j]):
                ops_names.append(ops_dict[j])
    return ops_names


def LC_graph_operations(z_1, solution):
    """
    Finds a list of operators needed to be applied on each qubit of the first graph to transform in to the second,
        given the Clifford transformation matrix, which is the output of the solver function.
    :param z_1: The adjacency matrix of the first graph. This is equal to the binary matrix for representing Pauli Z
        part of the symplectic binary representation of the stabilizer generators
    :type z_1: numpy.ndarray
    :param solution: an array of single-qubit Clifford 2*2 matrices in the symplectic formalism
    :type solution: numpy.ndarray
    :return: a list of the names of the operations that need to be applied on each qubit in the correct order.
    :rtype: list[str]
    """
    # takes an adjacency matrix and the solution (Clifford operation) and returns the list of local complementations
    # needed for graph transformation.
    r_matrix = _R_matrix(z_1, solution)
    n = np.shape(r_matrix)[0]
    singles_list = []
    doubles_list = []

    while _condition(r_matrix):
        r_matrix, s_list = _singles(r_matrix)
        singles_list.extend(s_list)

    while not np.array_equal(r_matrix, np.eye(n)):
        r_matrix, d_list = _doubles(r_matrix)
        doubles_list.extend(d_list)
    g_list = singles_list
    for i in doubles_list:
        g_list.append(i[0])
        g_list.append(i[1])
        g_list.append(i[0])
    return g_list


def iso_equal_check(G1, G2):
    """
    Checks if the graph G1 is LC equivalent to any graph that is isomorphic to G2
    :param G1: initial graph
    :type G1: networkx.Graph
    :param G2: target graph
    :type G2: networkx.Graph
    :return: If equivalent (True,the graph that G1 is equivalent to) and if not, (False, G1 itself)
    :rtype: tuple(bool, networkx.Graph)
    """
    iso_graphs_g2 = iso_graph_finder(G2)
    z_1 = nx.to_numpy_array(G1)
    iso_z_2 = [nx.to_numpy_array(G) for G in iso_graphs_g2]
    for x in iso_z_2:
        solution = solver(z_1, x)
        if isinstance(solution, type(np.array([0]))):
            return True, nx.to_networkx_graph(x)
        else:
            clear_output(wait=True)
    return False, G1


def iso_graph_finder(G):
    """
    Generates the list of all graphs that are isomorphic to the input graph G.
    :param G: input graph
    :type G: networkx.Graph
    :return: the list of graphs that are isomorphic to G
    :rtype: list[networkx.Graph]
    """
    iso_graphs = []
    list_nodes = sorted(G)
    n = len(list_nodes)
    permu = list(permutations(list_nodes, len(list_nodes)))

    for x in permu:
        adj_matrix = np.zeros((n, n))
        map_dict = dict(zip(list_nodes, x))
        g_copy = nx.relabel_nodes(G, map_dict, copy=True)
        for y in list_nodes:
            for z in list(g_copy.neighbors(y)):
                adj_matrix[y, z] = 1

        g_copy = nx.to_networkx_graph(adj_matrix)
        iso_graphs.append(g_copy)

    return iso_graphs


def local_comp_graph(G1, i):
    """
    Applies a local complementation on the i-th node of the input graph G1 and returns a copy.
    :param G1: input graph
    :type G1: networkx.Graph
    :param i: the index of the node to apply tho local complementation on.
    :type i: int
    :return: a new transformed graph
    :rtype: networkx.Graph
    """
    n = G1.number_of_nodes()
    assert (n > i >= 0), "node index is not in graph"
    z_1 = nx.to_numpy_array(G1).astype(int)
    identity = np.eye(n, n)
    gamma_matrix = np.zeros((n, n))
    gamma_matrix[i, i] = 1  # gamma has only a single 1 element on position = diag(i)
    z_2 = (z_1 @ (gamma_matrix @ z_1 + z_1[i, i] * gamma_matrix + identity) % 2) % 2
    for j in range(n):
        z_2[j, j] = 0
    g2 = nx.to_networkx_graph(z_2)
    plt.figure(1)
    nx.draw(G1, with_labels=True)
    plt.figure(2)
    nx.draw(g2, with_labels=True)
    return g2


def _R_matrix(z_1, solution):
    """
    R matrix calculator which is C * Z + D where C and D are the lower blocks of the total Clifford operator in the
        symplectic formalism. Each row of z_1 matrix (the i-th qubit's row) is multiplied by C_i and D_i is added to the
        diagonal element Z_ii which is zero by default!
    :param z_1: The adjacency matrix of the first graph. This is equal to the binary matrix for representing Pauli Z
        part of the symplectic binary representation of the stabilizer generators
    :type z_1: networkx.Graph
    :param solution: an array of single-qubit Clifford 2*2 matrices in the symplectic formalism
    :type solution: numpy.ndarray
    :return: the R matrix
    :rtype: numpy.ndarray
    """
    n = np.shape(z_1)[0]
    r_matrix = 0 * z_1
    for i in range(n):
        r_matrix[i] = solution[i, 1, 0] * z_1[i]  # the C*Z part. The element C_ii = solution[i,1,0]
        r_matrix[i, i] = solution[i, 1, 1]  # the D part. The element C_ii = solution[i,1,1]
    return r_matrix


def _apply_f(r_matrix, i):
    """
    Applies f_i transformation on the R matrix
    :param r_matrix: the R matrix = C * Z + D
    :type r_matrix: numpy.ndarray
    :param i: the index of the node or qubit on which the transformation is applied
    :type i: int
    :return: the transformed R matrix
    :rtype: numpy.ndarray
    """
    n = np.shape(r_matrix)[0]
    identity = np.eye(n, n)
    gamma_matrix = np.zeros((n, n))
    gamma_matrix[i, i] = 1
    r_matrix = (r_matrix @ (gamma_matrix @ r_matrix + r_matrix[i, i] * gamma_matrix + identity) % 2) % 2
    return r_matrix


def _singles(r_matrix):
    """
    Applies single f_i transformation on R matrix and records index "i" until no more single "f" transforms are needed.
    :param r_matrix: the R matrix = C * Z + D
    :type r_matrix: numpy.ndarray
    :return: the transformed R matrix, list of the indices on which f_i was applied
    :rtype: numpy.ndarray, list[int]
    """
    n = np.shape(r_matrix)[0]
    singles_list = []
    for i in range(n):
        if r_matrix[i, i] == 1 and (not np.array_equal(r_matrix[i], np.eye(n)[i])):
            singles_list.append(i)
            r_matrix = _apply_f(r_matrix, i)
    return r_matrix, singles_list


def _doubles(r_matrix):
    """
    Applies double f_ij transformation on R matrix and records index "i,j" until no more double "f"s are needed.
    :param r_matrix: the R matrix = C * Z + D
    :type r_matrix: numpy.ndarray
    :return: the transformed R matrix, list of the indices on which f_i was applied
    :rtype: numpy.ndarray, list[tuple(int, int)]
    """
    n = np.shape(r_matrix)[0]
    doubles_list = []
    for j in range(n):
        if not np.array_equal(r_matrix[j], np.eye(n)[j]) and r_matrix[j, j] == 0:
            k_list = []
            for k in range(n):
                if r_matrix[k, j] == 1:
                    k_list.append(k)
            r_matrix = _apply_f(r_matrix, j)
            r_matrix = _apply_f(r_matrix, k_list[0])
            r_matrix = _apply_f(r_matrix, j)
            doubles_list.append((j, k_list[0]))
    return r_matrix, doubles_list


def _condition(r_matrix):  #
    """
    Checks if further single "f" transformations are possible
    :param r_matrix: the R matrix = C * Z + D
    :type r_matrix: numpy.ndarray
    :return: True if more single "f"s are allowed and False otherwise.
    :rtype: bool
    """
    n = np.shape(r_matrix)[0]
    cond = False
    for i in range(n):
        cond = cond or (r_matrix[i, i] == 1 and (not (np.array_equal(r_matrix[i].astype(int), np.eye(n)[i]))))
    return cond


def _tester(n):
    """
    A test function for internal purpose. Searches over all random graphs of size "n" to finally find two that are
        LC equivalent. Should not be used for n > 7 since it may never find a solution in reasonable time.
    :param n: the size of the graphs (number of the nodes/ qubits)
    :type n: int
    :return: solution, G1, G2. Two graphs that are LC equivalent and the Clifford operator needed for the transformation
    :rtype: numpy.ndarray, networkx.Graph, networkx.Graph
    """
    solution = 1
    while not isinstance(solution, type(np.array([0]))):
        g1, g2 = nx.fast_gnp_random_graph(n, 0.65), nx.fast_gnp_random_graph(n, 0.65)

        z_1 = nx.to_numpy_array(g1).astype(int)
        z_2 = nx.to_numpy_array(g2).astype(int)

        solution = solver(z_1, z_2, Mode='deterministic')

        if isinstance(solution, type(np.array([0]))):
            plt.figure(1)
            nx.draw(g1, with_labels=True)
            plt.figure(2)
            nx.draw(g2, with_labels=True)
            print(local_clifford_ops(solution), "\n")
        else:
            clear_output(wait=True)

    return solution, (g1, g2)
