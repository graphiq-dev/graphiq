from functools import reduce

import pytest

import graphiq.backends.stabilizer.functions.clifford as sfc
import graphiq.backends.stabilizer.functions.metric as sfm
import graphiq.backends.stabilizer.functions.rep_conversion as conversion
import graphiq.backends.stabilizer.functions.utils as sfu
from benchmarks.circuits import *
from graphiq.backends.density_matrix.compiler import DensityMatrixCompiler
from graphiq.backends.stabilizer.compiler import StabilizerCompiler
from graphiq.backends.stabilizer.functions.stabilizer import rref
from graphiq.backends.stabilizer.tableau import StabilizerTableau
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
    r_vector1 = np.zeros([n_nodes, 1])
    r_vector2 = np.zeros([n_nodes, 1])
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


def test_qubit_insertion2():
    tableau = sfc.create_n_ket0_state(4)
    tableau = sfc.insert_qubit(tableau, 1)
    assert np.array_equal(tableau.table, sfc.create_n_ket0_state(5).table)


def test_qubit_removal():
    tableau = sfc.create_n_plus_state(4)
    tableau = sfc.remove_qubit(tableau, 1)


def test_qubit_insertion_removal():
    tableau = sfc.create_n_plus_state(4)
    tableau = sfc.insert_qubit(tableau, 1)
    sfc.remove_qubit(tableau, 1)


def test_stabilizer_inner_product():
    tableau1 = sfc.create_n_plus_state(4)
    tableau2 = sfc.create_n_ket0_state(4)
    assert sfm.inner_product(tableau1, tableau2) == 0.25
    assert sfm.inner_product(tableau1, tableau1) == 1.0


def test_stabilizer_fidelity():
    tableau1 = sfc.create_n_plus_state(4)
    tableau2 = sfc.create_n_ket0_state(4)
    tableau3 = sfc.create_n_ket1_state(4)
    assert sfm.fidelity(tableau1, tableau2) == 0.25 ** 2
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
    dm_compiler = DensityMatrixCompiler()
    dm_compiler.measurement_determinism = 1
    stab_compiler = StabilizerCompiler()
    stab_compiler.measurement_determinism = 1
    state1 = dm_compiler.compile(dag)
    state1.partial_trace([*range(3)], 4 * [2])
    print(metric.evaluate(state1, dag))
    state2 = stab_compiler.compile(dag)
    state2.partial_trace([*range(3)], 4 * [2])
    print(metric.evaluate(state2, dag))


def test_clifford_from_stabilizer():
    tableau1 = sfc.create_n_plus_state(4)
    tableau2 = conversion.clifford_from_stabilizer(tableau1.to_stabilizer())
    assert tableau1 == tableau2
    print(tableau1)
    print(tableau2)
