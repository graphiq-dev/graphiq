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
``MixedStabilizer.reduce`` folds branches that represent the same stabilizer
state, even when their tableaux differ: another route to the state, another
destabilizer completion or signs, or the generators in another order. Branches
of different states stay apart, the mixed state is unchanged, and the input
tableaux are not modified. The folding of tableaux that are ``==`` to each
other is covered in
tests/test_state_representation/stabilizer/test_mixed_stabilizer_reduce.py.
"""

import numpy as np

import graphiq.backends.stabilizer.functions.transformation as transform
import graphiq.backends.state_rep_conversion as rc
import graphiq.circuit.ops as ops
import graphiq.noise.noise_models as nm
from graphiq.backends.stabilizer.clifford_tableau import CliffordTableau
from graphiq.backends.stabilizer.compiler import StabilizerCompiler
from graphiq.backends.stabilizer.state import MixedStabilizer
from graphiq.circuit.circuit_dag import CircuitDAG


def _density(mixture):
    return sum(p * rc.stabilizer_to_density(t.to_stabilizer()) for p, t in mixture)


def _bell_routes():
    route_a = CliffordTableau(2)
    route_a = transform.hadamard_gate(route_a, 0)
    route_a = transform.cnot_gate(route_a, 0, 1)

    route_b = CliffordTableau(2)
    route_b = transform.hadamard_gate(route_b, 1)
    route_b = transform.cnot_gate(route_b, 1, 0)
    return route_a, route_b


def _row_swapped_zero_state():
    # the generators Z0, Z1 in the other order, with the destabilizer rows
    # X0, X1 swapped along with them so the tableau stays valid
    swapped_table = CliffordTableau(2).table.copy()
    swapped_table[[2, 3]] = swapped_table[[3, 2]]
    swapped_table[[0, 1]] = swapped_table[[1, 0]]
    return CliffordTableau(swapped_table)


def _noisy_hadamard_circuit(n_emitters, depolarizing_prob):
    circ = CircuitDAG(n_emitter=n_emitters, n_photon=0, n_classical=0)
    for i in range(n_emitters):
        circ.add(
            ops.Hadamard(
                register=i,
                reg_type="e",
                noise=nm.DepolarizingNoise(depolarizing_prob=depolarizing_prob),
            )
        )
    return circ


def _noisy_compiler():
    compiler = StabilizerCompiler()
    compiler.noise_simulation = True
    compiler.measurement_determinism = 1
    return compiler


def test_two_routes_to_the_same_state_reduce_to_one_branch():
    route_a, route_b = _bell_routes()
    assert route_a != route_b  # the tableaux themselves differ

    mixed = MixedStabilizer([(0.5, route_a), (0.5, route_b)])
    mixed.reduce()

    assert len(mixed.mixture) == 1
    probability, _ = mixed.mixture[0]
    assert probability == 1.0


def test_branches_differing_only_in_stabilizer_phase_stay_separate():
    # |00> and |10>: the same generators up to sign are different states
    zero_state = CliffordTableau(2)
    one_state = CliffordTableau(2)
    one_state.phase[one_state.n_qubits] = 1

    mixed = MixedStabilizer([(0.5, zero_state), (0.5, one_state)])
    mixed.reduce()

    assert len(mixed.mixture) == 2


def test_branches_differing_only_in_destabilizer_rows_merge():
    # stabilizers Z0, Z1 completed by the destabilizers X0Z1, X1Z0 instead of
    # X0, X1: another valid tableau of |00>
    same_state = CliffordTableau(2)
    other_completion = CliffordTableau(2)
    other_completion.destabilizer = np.array([[1, 0, 0, 1], [0, 1, 1, 0]])

    assert same_state != other_completion  # the tableaux themselves differ

    mixed = MixedStabilizer([(0.5, same_state), (0.5, other_completion)])
    mixed.reduce()

    assert len(mixed.mixture) == 1
    probability, _ = mixed.mixture[0]
    assert probability == 1.0


def test_branches_differing_only_in_destabilizer_signs_merge():
    # what a Pauli error does to a branch whose state it leaves alone: Z on |0>
    # flips the sign of the destabilizer X only
    plain = CliffordTableau(1)
    flipped = transform.z_gate(CliffordTableau(1), 0)
    assert plain != flipped  # the tableaux themselves differ

    mixed = MixedStabilizer([(0.75, plain), (0.25, flipped)])
    mixed.reduce()

    assert len(mixed.mixture) == 1
    probability, _ = mixed.mixture[0]
    assert probability == 1.0


def test_branches_differing_only_in_generator_row_order_merge():
    reference = CliffordTableau(2)
    reordered = _row_swapped_zero_state()
    assert reference != reordered  # the tableaux themselves differ

    mixed = MixedStabilizer([(0.5, reference), (0.5, reordered)])
    mixed.reduce()

    assert len(mixed.mixture) == 1
    probability, _ = mixed.mixture[0]
    assert probability == 1.0


def test_reduce_does_not_modify_any_input_tableau():
    # Bringing a branch to canonical form must not write back into it. The
    # stabilizer half of route_b and of the row-swapped tableau is not already
    # canonical, so a write-back would show on them; the signed copy of route_b
    # also shows a write-back through its phase vector.
    route_a, route_b = _bell_routes()
    signed_b = transform.z_gate(route_b.copy(), 0)
    inputs = [
        route_a,
        route_b,
        CliffordTableau(2),
        _row_swapped_zero_state(),
        signed_b,
    ]
    snapshots = [
        (np.copy(t.table), np.copy(t.phase), np.copy(t.iphase)) for t in inputs
    ]

    mixed = MixedStabilizer([(0.2, t) for t in inputs])
    mixed.reduce()

    assert len(mixed.mixture) == 3
    for t, (table, phase, iphase) in zip(inputs, snapshots):
        assert np.array_equal(t.table, table)
        assert np.array_equal(t.phase, phase)
        assert np.array_equal(t.iphase, iphase)


def test_reduce_preserves_total_probability_on_a_noisy_compile(monkeypatch):
    # a real noise-simulation mixture: folding must not drop or double-count
    # probability mass, and must leave the mixed state as it was
    circ = _noisy_hadamard_circuit(3, 0.1)

    monkeypatch.setattr(nm, "REDUCE_STABILIZER_MIXTURE", False)
    unreduced = _noisy_compiler().compile(circ).rep_data
    monkeypatch.setattr(nm, "REDUCE_STABILIZER_MIXTURE", True)
    reduced = _noisy_compiler().compile(circ).rep_data
    assert isinstance(unreduced, MixedStabilizer)
    assert isinstance(reduced, MixedStabilizer)

    total_unreduced = sum(p for p, _ in unreduced.mixture)
    total_reduced = sum(p for p, _ in reduced.mixture)
    assert np.isclose(total_unreduced, 1.0)
    assert np.isclose(total_reduced, total_unreduced)
    assert np.allclose(_density(reduced.mixture), _density(unreduced.mixture))


def test_reduce_leaves_the_mixed_state_unchanged(monkeypatch):
    monkeypatch.setattr(nm, "REDUCE_STABILIZER_MIXTURE", False)
    mixed = _noisy_compiler().compile(_noisy_hadamard_circuit(3, 0.1)).rep_data
    n_before = len(mixed.mixture)
    rho_before = _density(mixed.mixture)

    mixed.reduce()

    assert len(mixed.mixture) < n_before
    assert np.isclose(sum(p for p, _ in mixed.mixture), 1.0)
    assert np.allclose(_density(mixed.mixture), rho_before)


def test_noisy_compile_keeps_one_branch_per_distinct_state():
    # depolarizing noise after H on each of three emitters: 4^3 Kraus branches,
    # but only the 2^3 states |+/->^3
    mixed = _noisy_compiler().compile(_noisy_hadamard_circuit(3, 0.1)).rep_data

    assert len(mixed.mixture) == 8
    assert np.isclose(sum(p for p, _ in mixed.mixture), 1.0)


def test_mixture_over_every_two_qubit_tableau_folds_to_the_sixty_states():
    # breadth-first over Clifford gates from |00>, keeping every tableau reached
    # (many per state): folding must give exactly one branch per stabilizer state
    gates = [(transform.hadamard_gate, (q,)) for q in range(2)]
    gates += [(transform.phase_gate, (q,)) for q in range(2)]
    gates += [(transform.x_gate, (q,)) for q in range(2)]
    gates += [(transform.cnot_gate, (0, 1)), (transform.cnot_gate, (1, 0))]
    seen, frontier, tableaux = set(), [CliffordTableau(2)], []
    while frontier:
        next_frontier = []
        for t in frontier:
            key = (t.table.tobytes(), t.phase.tobytes())
            if key in seen:
                continue
            seen.add(key)
            tableaux.append(t)
            next_frontier.extend(gate(t.copy(), *args) for gate, args in gates)
        frontier = next_frontier

    mixed = MixedStabilizer([(1 / len(tableaux), t) for t in tableaux])
    rho = _density(mixed.mixture)
    mixed.reduce()

    assert len(mixed.mixture) == 60
    assert np.allclose(_density(mixed.mixture), rho)
