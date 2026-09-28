# Copyright (c) 2022-2024 Quantum Bridge Technologies Inc.
# Copyright (c) 2022-2024 Ki3 Photonics Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from functools import reduce

import pytest
import networkx as nx
import numpy as np
import graphiq.backends.stabilizer.functions.clifford as sfc
import graphiq.backends.stabilizer.functions.metric as sfm
import graphiq.backends.stabilizer.functions.stabilizer as sfs
import graphiq.backends.stabilizer.functions.rep_conversion as conversion
import graphiq.backends.stabilizer.functions.utils as sfu
from graphiq.circuit import ops
from graphiq.benchmarks.circuits import (
    linear_cluster_3qubit_circuit,
    linear_cluster_4qubit_circuit,
)
from graphiq.circuit.circuit_dag import CircuitDAG
from graphiq.backends.density_matrix.compiler import DensityMatrixCompiler
from graphiq.backends.stabilizer.compiler import StabilizerCompiler
from graphiq.backends.stabilizer.functions.stabilizer import rref
from graphiq.backends.stabilizer.tableau import StabilizerTableau
from graphiq.backends.state_rep_conversion import stabilizer_to_density
from graphiq.metrics import Infidelity


def test_symplectic_to_string():
    x_matrix1 = np.zeros((4, 4))
    z_matrix1 = np.eye(4)
    expected1 = ["ZIII", "IZII", "IIZI", "IIIZ"]
    assert reduce(
        lambda x, y: x and y,
        map(
            lambda a, b: a == b,
            sfu.symplectic_to_string(x_matrix1, z_matrix1),
            expected1,
        ),
        True,
    )

    x_matrix2 = np.array([[1, 0, 0, 0], [0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
    z_matrix2 = np.eye(4)
    expected2 = ["YIII", "IZII", "IXZI", "IIXZ"]
    assert reduce(
        lambda x, y: x and y,
        map(
            lambda a, b: a == b,
            sfu.symplectic_to_string(x_matrix2, z_matrix2),
            expected2,
        ),
        True,
    )


def test_string_to_symplectic():
    generator_list = ["ZIII", "IZII", "IIZI", "IIIZ"]
    x_matrix, z_matrix = sfu.string_to_symplectic(generator_list)
    assert np.array_equal(x_matrix, np.zeros((4, 4)))
    assert np.array_equal(z_matrix, np.eye(4))


def test_symplectic_product():
    adj_matrix = np.array(
        [[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]]
    ).astype(int)
    tableau = np.block([[np.eye(4), adj_matrix]])
    dim = 2

    assert np.array_equal(
        sfu.binary_symplectic_product(tableau, tableau), np.zeros((4, 4))
    )
    assert sfu.is_symplectic_self_orthogonal(tableau)


@pytest.mark.parametrize("n_nodes", [*range(5, 15)])
def test_echelon_form(n_nodes):
    g1 = nx.complete_graph(n_nodes)
    g2 = nx.gnp_random_graph(n_nodes, 0.5)
    r_vector1 = np.zeros(n_nodes)
    r_vector2 = np.zeros(n_nodes)
    z1 = nx.to_numpy_array(g1).astype(int)
    z2 = nx.to_numpy_array(g2).astype(int)
    x1 = np.eye(n_nodes)
    x2 = np.eye(n_nodes)
    tableau1 = StabilizerTableau([x1, z1], r_vector1)
    tableau2 = StabilizerTableau([x2, z2], r_vector2)
    tableau1 = rref(tableau1)
    tableau2 = rref(tableau2)
    x1 = tableau1.x_matrix
    z1 = tableau1.z_matrix
    x2 = tableau2.x_matrix
    z2 = tableau2.z_matrix
    pivot = [0, 0]
    # print("1", "\n", x1.astype(int), "\n", z1)
    for i in range(n_nodes):
        # pivot[1] = i
        old_pivot = pivot[0]
        nonzero_x1 = np.nonzero(x1[:, i])[0]
        nonzero_z1 = np.nonzero(z1[:, i])[0]
        if len(nonzero_x1) == 0 or len(nonzero_z1) == 0:
            if len(nonzero_x1) == 0 and len(nonzero_z1) == 0:
                pivot[0] = old_pivot
            elif len(nonzero_z1) == 0:
                pivot[0] = max(nonzero_x1[-1], old_pivot)
            else:
                pivot[0] = max(nonzero_z1[-1], old_pivot)
        else:
            pivot[0] = max(nonzero_x1[-1], nonzero_z1[-1], old_pivot)
        # print(f"pivot ={pivot}")
        assert pivot[0] - old_pivot <= 2
        for j in range(1 + int(pivot[0]), n_nodes):
            assert int(x1[j, i]) == 0 and int(z1[j, i]) == 0
    pivot = [0, 0]
    # print("2", "\n", x2.astype(int), "\n", z2)
    for i in range(n_nodes):
        # pivot[1] = i
        old_pivot = pivot[0]
        nonzero_x2 = np.nonzero(x2[:, i])[0]
        nonzero_z2 = np.nonzero(z2[:, i])[0]
        if len(nonzero_x2) == 0 or len(nonzero_z2) == 0:
            if len(nonzero_x2) == 0 and len(nonzero_z2) == 0:
                pivot[0] = old_pivot
            elif len(nonzero_z2) == 0:
                pivot[0] = max(nonzero_x2[-1], old_pivot)
            else:
                pivot[0] = max(nonzero_z2[-1], old_pivot)
        else:
            pivot[0] = max(nonzero_x2[-1], nonzero_z2[-1], old_pivot)
        # print(f"pivot ={pivot}")
        assert pivot[0] - old_pivot <= 2
        for j in range(int(pivot[0]) + 1, n_nodes):
            assert int(x2[j, i]) == 0 and int(z2[j, i]) == 0


def test_qubit_insertion():
    tableau = sfc.create_n_plus_state(4)
    tableau = sfc.insert_qubit(tableau, 1)
    assert tableau == sfc.tensor(
        [
            sfc.create_n_plus_state(1),
            sfc.create_n_ket0_state(1),
            sfc.create_n_plus_state(3),
        ]
    )


def test_qubit_insertion2():
    tableau = sfc.create_n_ket0_state(4)
    tableau = sfc.insert_qubit(tableau, 1)
    assert np.array_equal(tableau.table, sfc.create_n_ket0_state(5).table)


def test_qubit_removal():
    tableau = sfc.create_n_plus_state(4)
    tableau = sfc.remove_qubit(tableau, 1)
    # the measurement outcome is random, but every outcome leaves |+>^3
    assert tableau == sfc.create_n_plus_state(3)


def test_qubit_insertion_removal():
    tableau = sfc.create_n_plus_state(4)
    tableau = sfc.insert_qubit(tableau, 1)
    tableau = sfc.remove_qubit(tableau, 1)
    assert tableau == sfc.create_n_plus_state(4)


def test_stabilizer_inner_product():
    tableau1 = sfc.create_n_plus_state(4)
    tableau2 = sfc.create_n_ket0_state(4)
    assert sfm.inner_product(tableau1, tableau2) == 0.25
    assert sfm.inner_product(tableau1, tableau1) == 1.0


def test_stabilizer_fidelity():
    tableau1 = sfc.create_n_plus_state(4)
    tableau2 = sfc.create_n_ket0_state(4)
    tableau3 = sfc.create_n_ket1_state(4)
    assert sfm.fidelity(tableau1, tableau2) == 0.25**2
    assert sfm.fidelity(tableau1, tableau1) == 1.0
    assert sfm.fidelity(tableau2, tableau3) == 0.0


def test_stabilizer_fidelity2():
    dag = CircuitDAG(n_emitter=1, n_photon=3, n_classical=0)
    dag.add(
        ops.OneQubitGateWrapper([ops.Hadamard, ops.Phase], register=0, reg_type="e")
    )
    dag.add(ops.CNOT(control=0, control_type="e", target=0, target_type="p"))
    dag.add(ops.OneQubitGateWrapper([ops.Phase, ops.SigmaY], register=0, reg_type="p"))
    dag.add(ops.CNOT(control=0, control_type="e", target=1, target_type="p"))
    dag.add(ops.CNOT(control=0, control_type="e", target=2, target_type="p"))
    dag.add(ops.Hadamard(register=2, reg_type="p"))
    dag.add(
        ops.MeasurementCNOTandReset(
            control=0, control_type="e", target=1, target_type="p"
        )
    )
    dag.draw_circuit()
    _, target_state = linear_cluster_3qubit_circuit()
    metric = Infidelity(target_state)
    # the emitter measurement is random; both backends must agree on each outcome
    for determinism, expected in ((0, 0.75), (1, 1.0)):
        dm_compiler = DensityMatrixCompiler()
        dm_compiler.measurement_determinism = determinism
        stab_compiler = StabilizerCompiler()
        stab_compiler.measurement_determinism = determinism
        state1 = dm_compiler.compile(dag)
        state1.partial_trace([*range(3)], 4 * [2])
        infidelity1 = metric.evaluate(state1, dag)
        state2 = stab_compiler.compile(dag)
        state2.partial_trace([*range(3)], 4 * [2])
        infidelity2 = metric.evaluate(state2, dag)
        assert np.isclose(infidelity1, infidelity2)
        # outcome 0 leaves the photons at infidelity 0.75, outcome 1 at 1.0
        assert np.isclose(infidelity1, expected)


def test_clifford_from_stabilizer():
    tableau1 = sfc.create_n_plus_state(4)
    tableau2 = conversion.clifford_from_stabilizer(tableau1.to_stabilizer())
    assert tableau1 == tableau2
    print(tableau1)
    print(tableau2)


def _mixed_pauli_column():
    # first column holds X, Y, Z and I, one per row
    x_matrix = np.array([[1, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    z_matrix = np.array([[0, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]])
    return x_matrix, z_matrix


@pytest.mark.parametrize(
    "pauli_type, expected",
    [("x", [0]), ("y", [1]), ("z", [2]), ("X", [0]), ("Y", [1]), ("Z", [2])],
)
def test_one_pauli_type_finder_selects_the_rows_of_that_type(pauli_type, expected):
    x_matrix, z_matrix = _mixed_pauli_column()

    assert sfs.one_pauli_type_finder(x_matrix, z_matrix, [0, 0], pauli_type) == expected


@pytest.mark.parametrize("pauli_type", ["zz", "q", "i", "I", ""])
def test_one_pauli_type_finder_rejects_a_non_pauli_type(pauli_type):
    x_matrix, z_matrix = _mixed_pauli_column()

    with pytest.raises(ValueError, match=repr(pauli_type)):
        sfs.one_pauli_type_finder(x_matrix, z_matrix, [0, 0], pauli_type)


def test_the_row_scanners_are_bounded_by_the_row_count():
    # Four rows, two columns: a column-count bound stops after two rows and
    # never sees the Z in row 2.
    tall_x = np.array([[1, 0], [1, 0], [0, 0], [0, 0]])
    tall_z = np.array([[0, 0], [1, 0], [1, 0], [0, 0]])

    assert sfs.one_pauli_type_finder(tall_x, tall_z, [0, 0], "z") == [2]
    assert sfs.pauli_type_finder(tall_x, tall_z, [0, 0]) == ([0], [1], [2])

    # Two rows, four columns: a column-count bound indexes past the end.
    wide_x = np.array([[1, 0, 0, 0], [0, 0, 0, 0]])
    wide_z = np.array([[0, 0, 0, 0], [1, 0, 0, 0]])

    assert sfs.one_pauli_type_finder(wide_x, wide_z, [0, 0], "x") == [0]
    assert sfs.pauli_type_finder(wide_x, wide_z, [0, 0]) == ([0], [], [1])


@pytest.mark.parametrize(
    "shape", [(4, 2), (2, 4), (2,), (2, 2, 2)], ids=["tall", "wide", "1d", "3d"]
)
def test_stabilizer_tableau_rejects_a_non_square_pair(shape):
    with pytest.raises(ValueError, match="square 2-D"):
        StabilizerTableau([np.zeros(shape, dtype=int), np.zeros(shape, dtype=int)])


def test_stabilizer_to_density_rejects_too_few_generators():
    # two generators on three qubits used to give an (8, 8) matrix of trace 2
    with pytest.raises(ValueError, match="square 2-D"):
        stabilizer_to_density(["XZI", "ZXI"])


def test_stabilizer_tableau_accepts_a_square_pair():
    tableau = StabilizerTableau([np.eye(3, dtype=int), np.zeros((3, 3), dtype=int)])
    assert tableau.n_qubits == 3
    assert tableau.table.shape == (3, 6)


def test_is_symplectic_on_a_clifford_tableau():
    # a 2n x 2n Clifford table: P must be sized from the columns, not the rows
    n = 3
    assert sfu.is_symplectic(np.eye(2 * n, dtype=int))
    table = np.eye(2 * n, dtype=int)
    table[0, 1] = 1  # X_0 -> X_0 X_1, which anticommutes with Z_1: not symplectic
    assert not sfu.is_symplectic(table)


def test_binary_symplectic_product_of_different_row_counts():
    stabilizer = np.hstack([np.zeros((2, 2), dtype=int), np.eye(2, dtype=int)])  # ZI, IZ
    clifford = np.eye(4, dtype=int)  # XI, IX, ZI, IZ
    product = sfu.binary_symplectic_product(stabilizer, clifford)
    assert product.shape == (2, 4)
    # Z_i anticommutes with X_i only
    assert np.array_equal(product, np.array([[1, 0, 0, 0], [0, 1, 0, 0]]))
    assert sfu.is_symplectic_self_orthogonal(stabilizer)


@pytest.mark.parametrize(
    "z_rows",
    [
        # one identity generator: Z0, Z1, I
        [[1, 0, 0], [0, 1, 0], [0, 0, 0]],
        # dependent generators: Z0, Z1, Z0Z1 reduces to Z0, Z1, I
        [[1, 0, 0], [0, 1, 0], [1, 1, 0]],
        # the only generator on one qubit is the identity
        [[0]],
    ],
)
def test_rref_rejects_a_generator_that_reduces_to_the_identity(z_rows):
    z_matrix = np.array(z_rows)
    tableau = StabilizerTableau([np.zeros_like(z_matrix), z_matrix])
    with pytest.raises(AssertionError):
        rref(tableau)
