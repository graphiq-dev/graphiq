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
"""
``remove_qubit`` measures the traced-out qubit to disentangle it and then
deletes its columns. Applying X to the measured qubit on a 1 outcome puts +Z_q
in the group, at which point deleting the column transfers the outcome's sign
to the surviving generators. These tests hold that guarantee in both the
probabilistic and the deterministic branch, on small hand-checkable states: a
Bell pair, |110>, a case with more than one destabilizer row carrying the
removed qubit's X, and GHZ states.
"""

import numpy as np

import graphiq.backends.state_rep_conversion as rc
from graphiq.backends.stabilizer.functions.clifford import (
    create_n_ket0_state,
    partial_trace,
    remove_qubit,
    z_measurement_gate,
)
from graphiq.backends.stabilizer.functions.linalg import row_sum
from graphiq.backends.stabilizer.functions.transformation import (
    cnot_gate,
    hadamard_gate,
    x_gate,
)
from graphiq.backends.stabilizer.state import Stabilizer


def tableau_to_density(tableau):
    return rc.stabilizer_to_density(tableau.to_stabilizer())


def _bell_state():
    tableau = create_n_ket0_state(2)
    tableau = hadamard_gate(tableau, 0)
    tableau = cnot_gate(tableau, 0, 1)
    return tableau


def _combine_stabilizer_rows(tableau, row_to_add, target_row):
    """
    Overwrite one stabilizer generator with its product with another,
    in-place. Multiplying two generators together never changes the group
    they generate, only which basis of it is written down -- so this is a
    change of generator basis, not a change of state.

    :param tableau: the tableau to rewrite
    :type tableau: CliffordTableau
    :param row_to_add: stabilizer row index (0-based, within the stabilizer
        half) to multiply in
    :type row_to_add: int
    :param target_row: stabilizer row index (0-based, within the stabilizer
        half) to overwrite with the product
    :type target_row: int
    :return: the same tableau, rewritten
    :rtype: CliffordTableau
    """
    n_qubits = tableau.n_qubits
    (
        tableau.table_x,
        tableau.table_z,
        tableau.phase,
        tableau.iphase,
    ) = row_sum(
        tableau.table_x,
        tableau.table_z,
        tableau.phase,
        tableau.iphase,
        n_qubits + row_to_add,
        n_qubits + target_row,
    )
    return tableau


def test_removing_a_qubit_from_a_bell_pair_forced_to_outcome_zero_leaves_ket_0():
    tableau = remove_qubit(_bell_state(), 0, measurement_determinism=0)

    assert np.allclose(tableau_to_density(tableau), np.diag([1.0, 0.0]))


def test_removing_a_qubit_from_a_bell_pair_forced_to_outcome_one_leaves_ket_1():
    # This is the branch the un-fixed code gets wrong: outcome 1 never
    # transfers its sign to the surviving qubit, so the result reads back as
    # |0> instead of |1>.
    tableau = remove_qubit(_bell_state(), 0, measurement_determinism=1)

    assert np.allclose(tableau_to_density(tableau), np.diag([0.0, 1.0]))


def test_removing_a_qubit_from_ket_110_leaves_ket_10():
    # |110> is a product state, so the measurement of qubit 0 is deterministic
    # (x_p == 0), remove_qubit's *other* branch. Writing the state down in the
    # diagonal basis ['-ZII', '-IZI', '+IIZ'] would pass even with the defect,
    # because no surviving generator has support on qubit 0 to begin with, so
    # a row product first rewrites it as ['-ZII', '+ZZI', '+IIZ']: ZZI is the
    # surviving generator whose sign the un-fixed code drops.
    tableau = create_n_ket0_state(3)
    tableau = x_gate(tableau, 0)
    tableau = x_gate(tableau, 1)
    tableau = _combine_stabilizer_rows(tableau, row_to_add=0, target_row=1)

    tableau = remove_qubit(tableau, 0)

    expected = np.zeros((4, 4))
    expected[2, 2] = 1.0  # |10>
    assert np.allclose(tableau_to_density(tableau), expected)


def _ket_11_via_cnot():
    # CNOT(0, 1) on |10> gives |11>, but -- unlike building |11> with two X
    # gates -- it leaves *both* destabilizer rows carrying X on qubit 1: the
    # AG rule for CNOT(control=0, target=1) sends destabilizer X_0 -> X_0 X_1
    # while destabilizer X_1 is untouched, so column 1 of the destabilizer
    # block ends up nonzero on both rows. That is remove_qubit's
    # ``len(non_zero) > 1`` sub-branch (confirmed below), reached here while
    # the qubit being removed is deterministically |1>, so the sub-branch is
    # actually exercised with a sign to get right.
    tableau = create_n_ket0_state(2)
    tableau = x_gate(tableau, 0)
    tableau = cnot_gate(tableau, 0, 1)
    return tableau


def test_the_more_than_one_destabilizer_sub_branch_is_really_exercised():
    # A check on the fixture itself, not on remove_qubit: if this ever stopped
    # being true, the test below would still pass, but for the wrong reason
    # -- it would have quietly stopped covering the len(non_zero) > 1 branch.
    tableau = _ket_11_via_cnot()
    non_zero = [i for i in range(2) if tableau.destabilizer_x[i, 1] != 0]
    _, outcome, x_p = z_measurement_gate(tableau.copy(), 1)

    assert len(non_zero) > 1
    assert x_p == 0  # deterministic branch
    assert outcome == 1  # the branch that needs the sign fix


def test_removing_a_qubit_with_more_than_one_destabilizer_row_on_it():
    tableau = remove_qubit(_ket_11_via_cnot(), 1)

    assert np.allclose(tableau_to_density(tableau), np.diag([0.0, 1.0]))


def _ghz3():
    tableau = create_n_ket0_state(3)
    tableau = hadamard_gate(tableau, 0)
    tableau = cnot_gate(tableau, 0, 1)
    tableau = cnot_gate(tableau, 0, 2)
    return tableau


def test_removing_the_middle_qubit_of_a_ghz3_state_forced_to_zero_leaves_ket_00():
    # A single-qubit Z measurement of a GHZ state is probabilistic (x_p != 0)
    # but fully correlated: whichever outcome it lands on forces the other two
    # qubits to the same value, so the post-removal state is a product, not a
    # Bell pair.
    tableau = remove_qubit(_ghz3(), 1, measurement_determinism=0)

    assert np.allclose(tableau_to_density(tableau), np.diag([1.0, 0.0, 0.0, 0.0]))


def test_removing_the_middle_qubit_of_a_ghz3_state_forced_to_one_leaves_ket_11():
    # The un-fixed code gets this one wrong: it reads back as |00> too,
    # because outcome 1's sign never reaches the surviving generators.
    tableau = remove_qubit(_ghz3(), 1, measurement_determinism=1)

    assert np.allclose(tableau_to_density(tableau), np.diag([0.0, 0.0, 0.0, 1.0]))


def _ghz4():
    tableau = create_n_ket0_state(4)
    tableau = hadamard_gate(tableau, 0)
    tableau = cnot_gate(tableau, 0, 1)
    tableau = cnot_gate(tableau, 0, 2)
    tableau = cnot_gate(tableau, 0, 3)
    return tableau


def test_partial_trace_of_multiple_qubits_forced_to_one_leaves_ket_1():
    # sfc.partial_trace removes three qubits from a GHZ4 state in one call
    # (each a remove_qubit under the hood), which is the multi-qubit path
    # ~14 production call sites actually use. Forcing all three removed
    # measurements to outcome 1 must leave the surviving qubit at |1>.
    tableau = partial_trace(_ghz4(), keep=[0], dims=4 * [2], measurement_determinism=1)

    assert np.allclose(tableau_to_density(tableau), np.diag([0.0, 1.0]))


def test_stabilizer_partial_trace_accepts_measurement_determinism():
    # Stabilizer.partial_trace forwards measurement_determinism, so a caller
    # can pin the traced-out qubit's measurement. That matters because the
    # removal now transfers the outcome's sign to the surviving generators:
    # the outcome is observable in the result, where before it could not be,
    # so a caller with no way to force it would be left with a random state.
    state = Stabilizer(_ghz4())
    state.partial_trace(keep=[0], dims=4 * [2], measurement_determinism=1)

    assert np.allclose(tableau_to_density(state.tableau), np.diag([0.0, 1.0]))
