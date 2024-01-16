r"""
Local-Clifford equivalence check determines whether two graph states are local-Clifford equivalent via
:py:func:`is_lc_equivalent` function.
This section is based on articles [1]_ and [2]_; please refer to them for more details.

Any n-partite stabilizer state is defined by a set of :math:`n` commuting operators called the stabilizer generators.
Each member of this set is called a stabilizer, and consists of tensor product of local Pauli operators on each of the
qubits. The state is the unique common eigenstate of this set with eigenvalue equal to +1. The set itself is not
unique and any set of $n$ independent members of the stabilizer group, that is generated by this generator set,
is also valid. A binary representation of the stabilizers can be obtained by the following mapping:

$$
   \sigma_0 = I \rightarrow 00,\ \sigma_x \rightarrow 01,\  \sigma_z \rightarrow 10,\  \sigma_y \rightarrow 11
$$
This can be generalized to $n$-qubit case, which gives rise to a $2n$-dimensional vector.
For example, $\sigma_x \otimes \sigma_y \otimes \cdots \otimes \sigma_z
\rightarrow \left( 0, 1, \cdots, 1| 1, 1, \cdots, 0 \right)$.
We note that this convention is different from the one used by many other authors.
Nevertheless, as we treat $X$ and $Z$ matrices separately,
this different convention does not cause any issue.

The generators set can then be represented by $2n \times n$ matrix,
which can be thought of as two $n \times n$ square matrices on top of each other.
A column in this matrix is a $2n$-dimensional vector representing one stabilizer operator.
We name this matrix $S$, the top matrix $Z$ matrix, and the bottom matrix $X$ matrix.

$$
   S = \begin{bmatrix} Z \\X \end{bmatrix} = \begin{bmatrix} \theta \\ I \end{bmatrix}
$$

Under this mapping, if $Z_{ij}$ is equal to one and $X_{ij}$ is zero,
then it means a Pauli $Z$ operator acting on qubit $i$ in the $j$-th stabilizer operator.
The other way around which is $Z_{ij} = 0$ and $X_{ij} =1$ is
a Pauli X acting on qubit $i$. And if both elements are 0 or 1, they correspond to identity $I$ and
Pauli $Y$ operators, respectively.
For the special case of graph states, the $X$ matrix is equal to identity $I$
and the $Z$ matrix is the adjacency matrix of the graph usually shown by $\theta$.


A general $n$-qubit local Clifford operation in this formalism can be represented by
a $2n \times 2n$ matrix $Q$,which is made of 4 diagonal square matrices $A, B, C$ and $D$,
that is,
$$
   Q = \begin{bmatrix} A & B \\ C & D \end{bmatrix}.
$$

Each $2 \times 2$ sub-matrix, consisting of $i$-th diagonal elements of the $A, B, C, D$, acts locally
on the $i$-th qubit in the system, that is,

$$
   Q^{(i)} := \begin{bmatrix} a_{i} & b_{i}\\ c_{i} & d_{i} \end{bmatrix}.
$$

Considering two graph states with stabilizer generators $S = \begin{bmatrix} \theta \\ I \end{bmatrix}$ and
$S'= \begin{bmatrix} \theta' \\ I \end{bmatrix}$ the necessary and sufficient condition for LC
equivalence is the existence of the Clifford operator Q such that:

$$
    S^{T}Q^{T}PS' = 0
$$

where P is the $2n \times 2n$ matrix of the form  $P =\begin{bmatrix} 0 & I \\ I & 0 \end{bmatrix}$.
This gives rise to a system of $n^2$ linear equations to find $4n$ unknown elements of $Q$

$$
   \left(\sum_{i=1}^{n} \theta_{i j} \theta_{i k}^{\prime} c_{i}\right)+\theta_{j k} a_{k}+\theta_{j k}^{\prime} d_{
   j}+\delta_{j k} b_{j}=0
$$

The unknowns $a_{i},\  b_{i},\  c_{i},\  d_{i}$ should also satisfy the constraints

$$
    a_{i}d_{i} + b_{i}c_{i}=1
$$

to be a valid Clifford operation. (This ensures that the Clifford operation on a qubit is invertible.)

In order to solve for $Q$, one should first find a basis for the solution space of the system of linear equations.

$$
    B = \left\{ b_{1},\ldots,b_{d} \right\}
$$

Remember that if the system of equations is overdetermined after removing linearly dependent constraints,
then there exists no solution (other than :math:`Q=0` which is not a valid Clifford operation)
and graph states are not equivalent for sure. It is proved that a solution for a valid
$Q$ exists, if and only if the set

$$
    \left\{b+b^{\prime} \mid b, b^{\prime} \in B\right\}
$$

has a member that satisfies the constraint. Note that this is only true for the cases where $B$ is at least of size 4.
However, this is rarely the case and has not practical significance. For smaller $B$, one should test all possible
solutions (at most 16) to see if a valid $Q$ can be found.

In order to find a basis for set B, one needs to use the $n^2 \times 4n$ coefficient matrix of the system of linear
equations, find the linearly dependent columns (this is done in the code by a row-reduction process), and choose any
complete basis $B_{dependent}$ for that subspace, and use each member of that basis to form a set of new system of
linear equations.

$$
   \begin{align}
   A &x =& y \in B_{dependent} \\
   &x =& A^{-1} y = b \in B
   \end{align}
$$

Where the new full rank coefficient matrix $A$ is obtained from the old coefficient matrix after row-reduction and
removing the linearly dependent rows and columns, and the inhomogeneous vectors $y$ are the members of the basis
$B_{dependent}$.
Solving for vector $x$ will result in finding one of the members $b$ of solution basis $B$.
As evident here, the size of $B$ is equal to the size of $B_{dependent}$
which is equal to the number of linearly dependent columns in the initial coefficient matrix.

If $Q$ is found, the graphs are local-Clifford equivalent. In this case, one can translate the sub-matrices
$Q^{(i)}$ to local Clifford operations (combination of $H$ and $P$ gates)
needed (via :py:func:`local_clifford_ops` function) to convert one graph state to the other.

All general local-Clifford operations $Q$ on graph states are proved to be equivalent to a series of local
complementations on the graph. A list of local complementation graph operations can also be found (
:py:func:`lc_graph_operations` function) that transform one graph to the other one in a step by step graph transform.
Please read section IV of [2]_ for details. In order to do so one starts with
making a so-called :math:`R` matrix which is defined as

$$
    R = C\theta + D
$$

where $C$ and $D$ are the lower blocks of $Q$, the Clifford operator needed to transform the initial graph to the target
one, and $\theta$ is the adjacency matrix. The transformation $f_{i}$ on R is defined as $f_{i}(R)
= R(\Gamma_{i}R + X_{ii}\Gamma_{i} + I)$ and the transformation $f_{jk} := f_{j}f_{k}f_{j}$. It is proved that
if a sequence of transformations of $R$ exist such that

$$
    f_{j_{M}k_{M}} \cdots f_{j_{1}k_{1}} f_{i_{N}} \cdots f_{i_{1}} (R) = I
$$
then the same sequence of local complementations on corresponding graph nodes $j_{M}k_{M}$ and $i_{N}$
transforms the graph state the same as $Q$ does.

In order to find the sequence of "f" transformations, one checks for elements equal to 1 in the diagonal of the
$R$-matrix, if such element is found in $i$-th location and the corresponding row is not already equal to the canonical
unit vector $e_i = \left\{0, 0, \cdots, 1, 0, \cdots, 0 \right\}$ the $f_i$ transformation is applied on
$R$-matrix and added to the sequence. This is repeated until no such row is found. Then one searches for diagonal
elements equal to zero, and for each of them searches the corresponding column to find any element equal to 1,
suppose that the zero diagonal is at $j$-th location and the element = 1 in that column is at :math:`k`-th row.
Then a double $f_{jk}$ transformation is applied of $R$-matrix and added to the sequence.
This is repeated until no more eligible element can be found in the diagonal of the $R$-matrix.


.. [1] Maarten Van den Nest, Jeroen Dehaene, and Bart De Moor Phys. Rev. A 70, 034302 – Published 17 September 2004

.. [2] Maarten Van den Nest, Jeroen Dehaene, and Bart De Moor Phys. Rev. A 69, 022316 – Published 24 February 2004

"""

from itertools import combinations, permutations

import networkx as nx
import numpy as np

import graphiq.backends.stabilizer.functions.linalg as slinalg


def is_lc_equivalent(adj_matrix1, adj_matrix2, mode="deterministic", seed=0):
    r"""
    Determines whether two graph states are local-Clifford equivalent or not, given the adjacency matrices of the two.
    It takes two adjacency matrices as input and returns a numpy.ndarray containing $n$ ($2 \times 2$ arrays
    = clifford operations on each qubit.

    :param adj_matrix1: the adjacency matrix of the first graph. This is equal to the binary matrix for representing
        Pauli Z part of the symplectic binary representation of the stabilizer generators
    :type adj_matrix1: numpy.ndarray
    :param adj_matrix2: the adjacency matrix of the second graph
    :type adj_matrix2: numpy.ndarray
    :param mode: the chosen mode for finding solutions. It can be either 'deterministic' (default) or 'random'.
    :type mode: str
    :param seed: an optional input to set the random seed for the random search approach
    :type seed: int
    :raises AssertionError: if the number of rows in the row reduced matrix is less than the rank of coefficient matrix
        or if the number of linearly dependent columns is not equal to :math:`4n - rank`
        (for :math:`n` being the number of nodes in the graph)
    :return: If a solution is found, returns an array of single-qubit Clifford :math:`2 \\times 2` matrices
        in the symplectic formalism. If not, graphs are not LC equivalent and returns None.
    :rtype: bool, numpy.ndarray or None
    """

    n_nodes = np.shape(adj_matrix1)[0]
    # get the coefficient matrix for the system of linear equations
    coeff_matrix = _coeff_maker(adj_matrix1, adj_matrix2)

    # row reduction applied
    reduced_coeff_matrix, _, last_nonzero_row_index = slinalg.row_reduction(
        coeff_matrix, coeff_matrix * 0
    )

    rank = last_nonzero_row_index + 1
    # check for rank to see how many independent equations are there = rank of the matrix
    if rank >= 4 * n_nodes:
        # Those two graph states are not LC equivalent for sure
        return False, None
    # update the matrix to remove zero rows
    reduced_coeff_matrix = np.array([row for row in reduced_coeff_matrix if row.any()])
    assert (
        np.shape(reduced_coeff_matrix)[0] == rank
    ), "The number of remaining rows is less than the rank!"
    # rank = np.shape(reduced_coeff_matrix)[0]

    # finding linearly dependent columns
    col_list = _col_finder(reduced_coeff_matrix)
    length = len(col_list)
    assert length == 4 * n_nodes - rank, "column list is not correct"

    # if solution basis' length, which is the dimension of the solution basis, is less than 4 then we should check every
    # possible vector in solution basis:
    solution_basis = _solution_basis_finder(reduced_coeff_matrix, col_list)
    if len(solution_basis) < 5:
        basis_dimension = len(solution_basis)
        all_solutions = [*range(2**basis_dimension)]
        all_solutions = [list(format(i, f"0{basis_dimension}b")) for i in all_solutions]
        all_solutions = np.array(all_solutions).astype(int)
        all_solutions = all_solutions.T
        solution_basis = np.transpose(solution_basis, axes=(1, 2, 0)).reshape(
            4 * n_nodes, basis_dimension
        )
        all_solutions = (solution_basis @ all_solutions) % 2
        for solution in all_solutions.T:
            if _is_valid_clifford(solution.reshape(4 * n_nodes, 1)):
                # convert the solution to an array of n * (2 X 2) matrices
                valid_solution = solution.reshape(4 * n_nodes, 1)
                return True, valid_solution.reshape(n_nodes, 2, 2)
        return False, None

    if mode == "random":
        # Use random mode to get the fast convergence for large states
        # Check for random solutions 1000 times
        rand_solution = _random_checker(
            reduced_coeff_matrix, col_list, trial_count=1000, seed=seed
        )
        if isinstance(rand_solution, np.ndarray):
            # random result
            return True, rand_solution.reshape(n_nodes, 2, 2)
        else:
            return False, None

    elif mode == "deterministic":
        basis = _solution_basis_finder(reduced_coeff_matrix, col_list)
        possible_combinations = combinations(basis, 2)
        solution_set = []
        for basis_combination in possible_combinations:
            possible_solution = basis_combination[0] + basis_combination[1]
            solution_set.append(possible_solution % 2)
        for solution in solution_set:
            if _is_valid_clifford(solution):
                # convert the solution to an array of n * (2 X 2) matrices
                return True, solution.reshape(n_nodes, 2, 2)

        # states are NOT LC equivalent
        return False, None
    else:
        raise ValueError(
            'The mode should be either "random" or "deterministic" (default)'
        )


def _solution_basis_finder(reduced_coeff_matrix, col_list):
    """
    Finds the basis for all acceptable solutions in the system of linear equations.

    :param reduced_coeff_matrix: an echelon form matrix of the coefficients in the system of linear equations.
    :type reduced_coeff_matrix: numpy.ndarray
    :param col_list: list of the columns' indices in the coefficient matrix that are linearly dependent.
    :type col_list: list[int] or numpy.ndarray
    :raises AssertionError: if the solution basis found is wrong
    :return: an array containing the elements of the total Clifford operation needed to convert one graph to another.
    :rtype: numpy.ndarray
    """

    length = len(col_list)
    all_basis_elements = []
    for col in col_list:
        possible_basis = reduced_coeff_matrix[:, col]
        possible_basis = possible_basis.reshape(
            (np.shape(reduced_coeff_matrix)[1] - length, 1)
        )
        all_basis_elements.append(possible_basis)

    # contains all vectors b as its columns
    b_matrix = np.array(all_basis_elements)

    # removing linear dependent columns
    a_matrix = np.delete(reduced_coeff_matrix, col_list, axis=1)

    x_array = ((np.linalg.inv(a_matrix)) % 2 @ b_matrix) % 2
    x_array = x_array.astype(int)
    basis_list = np.eye(length).astype(int)
    llist = []
    for i in range(length):
        llist.append(list((x_array[i][:, 0])))
        for j in range(length):
            llist[i].insert(col_list[j], basis_list[i, j])
    v_array = np.array(llist)
    v_array = v_array.reshape((length, np.shape(reduced_coeff_matrix)[1], 1))

    # check also that it gives zero array as result:
    assert not ((reduced_coeff_matrix @ v_array) % 2).any(), "solution basis is wrong."

    return v_array.astype(int)


def _random_checker(reduced_coeff_matrix, col_list, trial_count=10000, seed=0):
    """
    Randomly searches for solutions to the system of linear equations for finding the Clifford operation matrix elements.
    This approach can be faster than the deterministic one for large states.
    However, it does not guarantee to produce a solution.

    :param reduced_coeff_matrix: an echelon form matrix of the coefficients in the system of linear equations.
    :type reduced_coeff_matrix: numpy.ndarray
    :param col_list: list of the columns' indices in the coefficient matrix that are linearly dependent.
    :type col_list: list[int] or numpy.ndarray
    :param trial_count: the number of trials
    :type trial_count: int
    :param seed: set the random seed for the random search approach
    :type seed: int
    :return: if successful, an array containing the elements of the total Clifford operation needed to convert one graph
        to another. If not, returns None.
    :rtype: numpy.ndarray or None
    """
    n = int(np.shape(reduced_coeff_matrix)[1] / 4)
    length = len(col_list)
    np.random.seed(seed)

    # rand_var_vec = a random choice of the variables' vector for the n-rank parameters that the equations cannot
    # handle!

    for j in range(trial_count):
        rand_var_vec = np.zeros((4 * n, 1))
        for i in range(length):
            rand_var_vec[col_list[i]] = np.random.randint(2)
        solution = _vec_solution_finder(reduced_coeff_matrix, col_list, rand_var_vec)
        if _is_valid_clifford(solution):
            # Random search is successful; return the solution found
            return solution
    # Random search is unsuccessful.
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
    # find the matrix equation Ax=b where A is the square (rank X rank) matrix out of the reduced_coeff_matrix x is the
    # vector of length=rank to be found by A^(-1)*b.
    # b is the vector obtained from randomly choosing the extra
    # unknowns of vector rand_vec that makes the non-homogeneous part of the
    # system of linear equations; b= (-1)* reduced_coeff_matrix * var_vec

    n = int(np.shape(reduced_coeff_matrix)[1] / 4)
    # removing linearly dependent columns
    a_square_reduced_coeff_matrix = np.delete(reduced_coeff_matrix, col_list, axis=1)

    b_nonhomogeneous = (reduced_coeff_matrix @ var_vec) % 2
    x_unknown_part_of_a_basis_vector = (
        (np.linalg.inv(a_square_reduced_coeff_matrix)) % 2 @ b_nonhomogeneous
    ) % 2

    # the full var_vec is now the x vector inserted to the var_vec vector to make all 4*n elements
    counter = 0
    for i in range(4 * n):
        if i not in col_list:
            var_vec[i] = x_unknown_part_of_a_basis_vector[i - counter][0]
        else:
            counter = counter + 1

    return var_vec.astype(int)


def _is_valid_clifford(vector):
    """
    Verifies if the given matrix elements for a Clifford operation is valid (if the Clifford matrix is invertible)。

    :param vector: an array containing the matrix elements of the total Clifford operation needed to convert one graph
        to another
    :type vector: numpy.ndarray
    :return: True if the input is valid and False if it is not
    :rtype: bool
    """
    # reshapes a 4*n vector into an array of 2*2 matrices which are the a_i, b_i, c_i, d_i  elements in Q = Clifford
    n = int(np.shape(vector)[0] / 4)
    vector_reshaped = vector.reshape(n, 2, 2)
    checklist = []
    for i in range(n):
        determinant_of_clifford = (
            vector_reshaped[i][0, 0] * vector_reshaped[i][1, 1]
        ) + (vector_reshaped[i][0, 1] * vector_reshaped[i][1, 0])
        checklist.append(int(determinant_of_clifford % 2))
    return all(checklist)


def _coeff_maker(z1_matrix, z2_matrix):
    """
    Forms the coefficient matrix for the system of linear equations to find the matrix elements of the Clifford
        operation needed to convert the initial graph to the target graph, given the adjacency matrices of the two.

    :param z1_matrix: The adjacency matrix of the first graph. This is equal to the binary matrix for representing
        Pauli Z part of the symplectic binary representation of the stabilizer generators
    :type z1_matrix: numpy.ndarray
    :param z2_matrix:The adjacency matrix of the second graph
    :type z2_matrix: numpy.ndarray
    :raises AssertionError: if two graphs have different numbers of nodes
    :return: the coefficient matrix for the system of linear equations for the Clifford operation matrix elements
    :rtype: numpy.ndarray
    """
    # z1 and z2 are initial and target adjacency matrices.
    # Returns the coefficient matrix for system of n**2 linear equations.
    n_nodes = np.shape(z1_matrix)[0]
    assert n_nodes == np.shape(z2_matrix)[0], "graphs must be of same size"

    coeff_matrix = np.zeros((n_nodes**2, 4 * n_nodes)).astype(int)
    for j in range(n_nodes):
        for k in range(n_nodes):
            for m in range(n_nodes):
                row = n_nodes * j + k
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


def _col_finder(row_reduced_coeff_matrix):
    """
    Finds linearly dependent columns in a row reduced matrix.

    :param row_reduced_coeff_matrix: the row reduced coefficient matrix
    :type row_reduced_coeff_matrix: numpy.ndarray
    :raises ValueError: if the input row_reduced_coeff_matrix is not a binary matrix
    :return: a list of the indices of the columns which are not linearly independent
    :rtype: numpy.ndarray
    """
    dependent_columns = []
    pivot = [0, 0]
    n_row, n_column = np.shape(row_reduced_coeff_matrix)
    for i in range(n_column - 1):
        if row_reduced_coeff_matrix[pivot[0], pivot[1]] == 1:
            if pivot[0] == (n_row - 1):
                pivot = [pivot[0], pivot[1] + 1]
                dependent_columns.extend([*range(pivot[1], n_column)])
                break
            else:
                pivot = [pivot[0] + 1, pivot[1] + 1]

        elif row_reduced_coeff_matrix[pivot[0], pivot[1]] == 0:
            dependent_columns.append(pivot[1])
            pivot = [pivot[0], pivot[1] + 1]
        else:
            raise ValueError("elements of matrix should be 0 or 1 only")
    return dependent_columns


def local_clifford_ops(solution):
    """
    Finds a list of operators needed to be applied on each qubit of the first graph to transform in to the second,
    given the Clifford transformation matrix, which is the output of the solver function.

    :param solution: an array of single-qubit Clifford :math:`2 \\times 2` matrices in the symplectic formalism
    :type solution: numpy.ndarray
    :return: a list of the names of the operations that need to be applied on each qubit in the correct order.
    :rtype: list[str]
    """
    # The order of the operations is the same as the qubits' labels in the graphs

    # allowed operations on single qubits in binary symplectic representation
    identity = np.array([[1, 0], [0, 1]])
    hadamard = np.array([[0, 1], [1, 0]])
    p = np.array([[1, 1], [0, 1]])
    ph = np.array([[1, 1], [1, 0]])
    hp_dagger = np.array([[0, 1], [1, 1]])
    php = np.array([[1, 0], [1, 1]])

    ops_list = [identity, hadamard, p, ph, hp_dagger, php]
    ops_list_str = ["I", "H", "P", "P H", "H P_dag", "P H P"]
    ops_dict = zip(list(range(len(ops_list))), ops_list_str)
    ops_dict = dict(ops_dict)
    ops_names = []
    for i in solution:
        for j in range(len(ops_list)):
            if np.array_equal(i, ops_list[j]):
                ops_names.append(ops_dict[j])
    return ops_names


def lc_graph_operations(adj_matrix, solution):
    """
    Finds a list of local complementations needed to be applied on each node of the given graph
    to transform it into the target graph given the Clifford transformation matrix,
    which is the output of the solver function.

    :param adj_matrix: The adjacency matrix of the first graph. This is equal to the binary matrix for
        representing Pauli Z part of the symplectic binary representation of the stabilizer generators
    :type adj_matrix: numpy.ndarray
    :param solution: an array of single-qubit Clifford :math:`2 \\times 2` matrices in the symplectic formalism
    :type solution: numpy.ndarray
    :return: a list of the names of the operations that need to be applied on each qubit in the correct order.
    :rtype: list[str]
    """
    # takes an adjacency matrix and the solution (Clifford operation) and returns the list of local complementations
    # needed for graph transformation.
    r_matrix = _R_matrix(adj_matrix, solution)
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


def find_lc_operations(adj_matrix1, adj_matrix2, mode="deterministic", seed=0):
    equivalency, solution = is_lc_equivalent(adj_matrix1, adj_matrix2, mode, seed)
    if equivalency:
        op_list = lc_graph_operations(adj_matrix2, solution)
        return op_list
    else:
        raise ValueError("These two graphs are not local-Clifford equivalent.")


def iso_equal_check(graph1, graph2):
    """
    Checks if the graph1 is local-Clifford equivalent to any graph that is isomorphic to graph2

    :param graph1: initial graph
    :type graph1: networkx.Graph
    :param graph2: target graph
    :type graph2: networkx.Graph
    :return: If equivalent (True,the graph that graph1 is equivalent to) and if not, (False, graph1 itself)
    :rtype: (bool, networkx.Graph)
    """
    iso_graphs_g2 = iso_graph_finder(graph2)
    adj_matrix1 = nx.to_numpy_array(graph1)
    iso_g2_adj_matrices = [nx.to_numpy_array(graph) for graph in iso_graphs_g2]
    for adj_matrix2 in iso_g2_adj_matrices:
        success, solution = is_lc_equivalent(adj_matrix1, adj_matrix2)
        if isinstance(solution, np.ndarray):
            return True, nx.to_networkx_graph(adj_matrix2)

    return False, graph1


def iso_graph_finder(input_graph):
    """
    Generates the list of all graphs that are isomorphic to the input graph G.
    Scales with n! and faces runtime or memory issues for large graphs.

    :param input_graph: input graph
    :type input_graph: networkx.Graph
    :return: the list of graphs that are isomorphic to G
    :rtype: list[networkx.Graph]
    """
    iso_graphs = []
    list_nodes = sorted(input_graph)
    n_nodes = len(list_nodes)
    all_permu = list(permutations(list_nodes, len(list_nodes)))

    for each_permu in all_permu:
        adj_matrix = np.zeros((n_nodes, n_nodes))
        map_dict = dict(zip(list_nodes, each_permu))
        g_copy = nx.relabel_nodes(input_graph, map_dict, copy=True)
        for node_id1 in list_nodes:
            for node_id2 in list(g_copy.neighbors(node_id1)):
                adj_matrix[node_id1, node_id2] = 1

        g_copy = nx.to_networkx_graph(adj_matrix)
        iso_graphs.append(g_copy)

    return iso_graphs


def local_comp_graph(input_graph, node_id):
    """
    Applies a local complementation on the node (specified by node_id) of the input graph input_graph
    and returns the resulting graph.

    :param input_graph: input graph
    :type input_graph: networkx.Graph
    :param node_id: the index of the node to apply tho local complementation on
    :type node_id: int
    :raises AssertionError: if the node index node_id is not in the graph
    :return: a new transformed graph
    :rtype: networkx.Graph
    """
    # TODO: integrate this with the state representation functions
    n_nodes = input_graph.number_of_nodes()
    assert n_nodes > node_id >= 0, "node index is not in the graph"
    adj_matrix = nx.to_numpy_array(input_graph).astype(int)
    identity = np.eye(n_nodes, n_nodes)
    gamma_matrix = np.zeros((n_nodes, n_nodes))

    # gamma has only a single 1 element on position = diag(i)
    gamma_matrix[node_id, node_id] = 1

    new_adj_matrix = (
        adj_matrix
        @ (
            gamma_matrix @ adj_matrix
            + adj_matrix[node_id, node_id] * gamma_matrix
            + identity
        )
        % 2
    ) % 2
    for j in range(n_nodes):
        new_adj_matrix[j, j] = 0
    new_graph = nx.to_networkx_graph(new_adj_matrix)

    return new_graph


def _R_matrix(adj_matrix, solution):
    """
    :math:`R` matrix calculator which is :math:`C \\times Z + D`
    where :math:`C` and :math:`D` are the lower blocks of the total Clifford operator in the symplectic formalism.
    Each row of z_1 matrix (the i-th qubit's row) is multiplied by :math:`C_i` and :math:`D_i` is added to the
    diagonal element :math:`Z_{ii}` which is zero by default.

    :param adj_matrix: The adjacency matrix of the first graph.
        This is equal to the binary matrix for representing Pauli Z
        part of the symplectic binary representation of the stabilizer generators
    :type adj_matrix: numpy.ndarray
    :param solution: an array of single-qubit Clifford :math:`2 \\times 2` matrices in the symplectic formalism
    :type solution: numpy.ndarray
    :return: the R matrix
    :rtype: numpy.ndarray
    """
    n_nodes = np.shape(adj_matrix)[0]
    r_matrix = 0 * adj_matrix
    for i in range(n_nodes):
        # the C*Z part. The element C_ii = solution[i,1,0]
        r_matrix[i] = solution[i, 1, 0] * adj_matrix[i]

        # the D part. The element C_ii = solution[i,1,1]
        r_matrix[i, i] = solution[i, 1, 1]
    return r_matrix


def _apply_f(r_matrix, i):
    """
    Applies :math:`f_i` transformation on the :math:`R` matrix

    :param r_matrix: the R matrix such that :math:`R = C \\times Z + D`
    :type r_matrix: numpy.ndarray
    :param i: the index of the node or qubit on which the transformation is applied
    :type i: int
    :return: the transformed R matrix
    :rtype: numpy.ndarray
    """
    n_nodes = np.shape(r_matrix)[0]
    identity = np.eye(n_nodes, n_nodes)
    gamma_matrix = np.zeros((n_nodes, n_nodes))
    gamma_matrix[i, i] = 1
    r_matrix = (
        r_matrix
        @ (gamma_matrix @ r_matrix + r_matrix[i, i] * gamma_matrix + identity)
        % 2
    ) % 2
    return r_matrix


def _singles(r_matrix):
    """
    Applies single :math:`f_i` transformation on R matrix and records index :math:`i` until no more single :math:`f`
    transforms are needed.

    :param r_matrix: the :math:`R` matrix such that :math:` R = C \\times Z + D`
    :type r_matrix: numpy.ndarray
    :return: the transformed :math:`R` matrix, list of the indices on which :math:`f_i` was applied
    :rtype: numpy.ndarray, list[int]
    """
    n_nodes = np.shape(r_matrix)[0]
    singles_list = []
    for i in range(n_nodes):
        if r_matrix[i, i] == 1 and (
            not np.array_equal(r_matrix[i], np.eye(n_nodes)[i])
        ):
            singles_list.append(i)
            r_matrix = _apply_f(r_matrix, i)
    return r_matrix, singles_list


def _doubles(r_matrix):
    """
    Applies double :math:`f_{ij}` transformation on R matrix and records index "i,j" until no more double "f"s are needed.

    :param r_matrix: the :math:`R` matrix so that :math:`R = C \\times Z + D`
    :type r_matrix: numpy.ndarray
    :return: the transformed :math:`R` matrix, list of the indices on which :math:`f_i` was applied
    :rtype: numpy.ndarray, list[(int, int)]
    """
    n_nodes = np.shape(r_matrix)[0]
    doubles_list = []
    for j in range(n_nodes):
        if not np.array_equal(r_matrix[j], np.eye(n_nodes)[j]) and r_matrix[j, j] == 0:
            k_list = []
            for k in range(n_nodes):
                if r_matrix[k, j] == 1:
                    k_list.append(k)
            r_matrix = _apply_f(r_matrix, j)
            r_matrix = _apply_f(r_matrix, k_list[0])
            r_matrix = _apply_f(r_matrix, j)
            doubles_list.append((j, k_list[0]))
    return r_matrix, doubles_list


def _condition(r_matrix):
    """
    Checks if further single :math:`f` transformations are possible

    :param r_matrix: the :math:`R` matrix so that :math:`R = C \\times Z + D`
    :type r_matrix: numpy.ndarray
    :return: True if more single :math:`f`s are allowed and False otherwise.
    :rtype: bool
    """
    n_nodes = np.shape(r_matrix)[0]
    cond = False
    for i in range(n_nodes):
        cond = cond or (
            r_matrix[i, i] == 1
            and (not (np.array_equal(r_matrix[i].astype(int), np.eye(n_nodes)[i])))
        )
    return cond
