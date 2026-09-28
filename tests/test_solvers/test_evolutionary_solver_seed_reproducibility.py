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
`SolverBase.seed` reseeds only `numpy.random` and the `random` module.
`_select_possible_cnot_position` and `_select_possible_measurement_position` used to build their
candidate list by iterating a Python `set` of edge tuples, whose iteration order depends on
`PYTHONHASHSEED`. Since CPython randomises that per process, the same numpy seed could select a
different edge in a different process, so a seeded search was not reproducible across processes.

Two properties are pinned here. The candidate order must not depend on `PYTHONHASHSEED`, and the
helpers must not order-compare node ids at all: ids are a mix of `str` (register endpoints) and
`int` (interior ids minted by `CircuitDAG._unique_node_id`), so `sorted()` raises
`TypeError: '<' not supported between instances of 'str' and 'int'`. Any fixture used here must
therefore contain inserted gates -- an *empty* `CircuitDAG` has only string ids and is exactly the
case in which a `sorted()`-based fix appears to work.

`PYTHONHASHSEED` is read once at interpreter startup, so this needs separate processes; it cannot
be exercised by forking or by reseeding in-process. Fingerprints use op class name, `register`,
`reg_type` and `q_registers` -- never `repr`/`str` of a circuit or op, which embeds a
`<... object at 0x...>` address that differs between processes regardless of any bug.
"""

import os
import pathlib
import subprocess
import sys

from graphiq.circuit import ops
from graphiq.circuit.circuit_dag import CircuitDAG
from graphiq.solvers.evolutionary_solver import EvolutionarySolver

# Several hash seeds, not two, because how many seeds it takes to expose the bug is
# fixture-dependent and not obvious from the fixture. Measured against the pre-fix helpers on
# the circuit `_CANDIDATE_ORDER_SCRIPT` builds, these five seeds produce five *distinct*
# candidate orders -- so any pair of them would do here. That is not something to rely on: on
# the smaller `_emitter_only_circuit` the same range collapses to two orders, and a two-seed
# comparison is then one coincidence away from passing against the live defect.
_HASH_SEEDS = (0, 1, 2, 3, 5)

# Searches are the slow path (~10s per process), so they span three seeds rather than five.
_SEARCH_HASH_SEEDS = (0, 1, 2)


def _emitter_only_circuit():
    """
    Mixed `str`/`int` node ids on the emitter edges, which is what
    `_select_possible_cnot_position` scans. Yields 4 candidate pairs.
    """
    circuit = CircuitDAG(n_emitter=2, n_photon=3, n_classical=1)
    circuit.add(ops.Hadamard(register=0, reg_type="e"))
    circuit.add(ops.CNOT(control=0, control_type="e", target=1, target_type="e"))
    circuit.add(ops.Hadamard(register=0, reg_type="e"))
    return circuit


def _emitter_and_photon_circuit():
    """
    Mixed ids on the photon edges too, so `_select_possible_measurement_position` returns a
    non-empty list. On `_emitter_only_circuit` that helper returns [] -- its inner loop never
    runs, so a `sorted()` call there would not raise and a test built on it would pass vacuously.
    """
    circuit = CircuitDAG(n_emitter=2, n_photon=3, n_classical=1)
    circuit.add(ops.Hadamard(register=0, reg_type="e"))
    circuit.add(ops.CNOT(control=0, control_type="e", target=0, target_type="p"))
    circuit.add(ops.CNOT(control=1, control_type="e", target=1, target_type="p"))
    circuit.add(ops.Hadamard(register=0, reg_type="p"))
    circuit.add(ops.CNOT(control=0, control_type="e", target=1, target_type="e"))
    return circuit


def _run_with_hashseed(script, hashseed, timeout=180):
    """
    Run `script` in a fresh subprocess under `PYTHONHASHSEED=hashseed`.

    A fork cannot express this: CPython reads `PYTHONHASHSEED` once at interpreter startup, so a
    forked child keeps the parent's hash seed whatever the environment says afterwards.
    """
    env = {**os.environ, "PYTHONHASHSEED": str(hashseed)}
    # Run from the repo root so the child imports this checkout's graphiq even when it is not
    # installed. Using os.getcwd() would make these tests pass only when pytest happens to be
    # run from the root.
    repo_root = pathlib.Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    assert (
        result.returncode == 0
    ), f"subprocess with PYTHONHASHSEED={hashseed} failed:\n{result.stderr}"
    return result.stdout.strip()


def test_node_ids_are_mixed_str_and_int():
    """
    The premise the other tests rest on. If node ids ever became uniformly typed this would fail,
    and the `sorted()` formulation these tests exclude would become viable -- so the exclusion
    should be revisited rather than silently kept.
    """
    circuit = _emitter_and_photon_circuit()

    id_types = {
        type(node).__name__
        for edge in circuit.edge_dict["e"] + circuit.edge_dict["p"]
        for node in edge[:2]
    }

    assert id_types == {"str", "int"}


def test_cnot_position_candidates_do_not_order_compare_node_ids():
    """
    `sorted()` on the set difference raises `TypeError` on this fixture. Asserting the exact
    length matters: an empty result would let this pass without the ordering code ever running.

    Note what this does NOT guard: it passes against the pre-fix helper too, which returns the
    same four pairs in a hash-dependent order. It pins that no *future* fix reintroduces an
    order comparison between mixed-type node ids -- the `sorted()` formulation this file's
    module docstring rules out -- not the set-iteration bug itself. The hash-seed tests below
    are the regression guards for that.
    """
    edge_pair = EvolutionarySolver._select_possible_cnot_position(
        _emitter_only_circuit()
    )

    assert len(edge_pair) == 4


def test_measurement_position_candidates_do_not_order_compare_node_ids():
    edge_pair = EvolutionarySolver._select_possible_measurement_position(
        _emitter_and_photon_circuit()
    )

    assert len(edge_pair) == 7


_CANDIDATE_ORDER_SCRIPT = """
from graphiq.circuit import ops
from graphiq.circuit.circuit_dag import CircuitDAG
from graphiq.solvers.evolutionary_solver import EvolutionarySolver

circuit = CircuitDAG(n_emitter=2, n_photon=3, n_classical=1)
circuit.add(ops.Hadamard(register=0, reg_type="e"))
circuit.add(ops.CNOT(control=0, control_type="e", target=0, target_type="p"))
circuit.add(ops.CNOT(control=1, control_type="e", target=1, target_type="p"))
circuit.add(ops.Hadamard(register=0, reg_type="p"))
circuit.add(ops.CNOT(control=0, control_type="e", target=1, target_type="e"))

print(repr(EvolutionarySolver._select_possible_cnot_position(circuit)))
print(repr(EvolutionarySolver._select_possible_measurement_position(circuit)))
"""


def test_candidate_order_is_hash_seed_independent():
    """
    The most direct pin of the defect: the helpers' own output on a fixed circuit, rather than a
    downstream search symptom. Covers both helpers in one process per seed.
    """
    orders = {
        seed: _run_with_hashseed(_CANDIDATE_ORDER_SCRIPT, seed) for seed in _HASH_SEEDS
    }

    assert len(set(orders.values())) == 1, (
        f"candidate order varies with PYTHONHASHSEED: {len(set(orders.values()))} "
        f"distinct orders across seeds {list(orders)}"
    )


_SEARCH_SCRIPT = """
import networkx as nx

from graphiq.backends.stabilizer.compiler import StabilizerCompiler
from graphiq.backends.stabilizer.functions.rep_conversion import (
    get_clifford_tableau_from_graph,
)
from graphiq.metrics import Infidelity
from graphiq.solvers.evolutionary_solver import EvolutionarySolver
from graphiq.state import QuantumState


def fingerprint(op):
    return (
        type(op).__name__,
        getattr(op, "register", None),
        getattr(op, "reg_type", None),
        op.q_registers,
    )


graph = nx.Graph({graph_edges})
target = QuantumState(get_clifford_tableau_from_graph(graph), rep_type="s")
compiler = StabilizerCompiler()
compiler.measurement_determinism = 1

solver = EvolutionarySolver(
    target=target,
    metric=Infidelity(target),
    compiler=compiler,
    n_emitter={n_emitter},
    n_photon={n_photon},
)
solver.seed({seed})
solver.solve()

score, circuit = solver.hof[0]
print(repr((score, [fingerprint(op) for op in circuit.sequence(unwrapped=True)])))
"""


def test_two_emitter_search_reproduces_across_hash_seeds():
    """
    At `n_emitter > 1` `add_emitter_cnot` is in `trans_probs`, so the search reaches
    `_select_possible_cnot_position`. Both the score and the winning circuit must match across
    hash seeds.
    """
    script = _SEARCH_SCRIPT.format(
        graph_edges="[(1, 2), (2, 3), (2, 4), (4, 3), (1, 3)]",
        n_emitter=2,
        n_photon=4,
        seed=3,
    )

    winners = {seed: _run_with_hashseed(script, seed) for seed in _SEARCH_HASH_SEEDS}

    assert len(set(winners.values())) == 1, (
        f"n_emitter=2 search gave {len(set(winners.values()))} distinct (score, circuit) "
        f"results across hash seeds {list(winners)} at one numpy seed"
    )


def test_one_emitter_search_is_hash_seed_independent():
    """
    At `n_emitter = 1` neither helper is reached -- `add_emitter_cnot` is gated on
    `n_emitter > 1` and `add_measurement_cnot_and_reset` is not in the default `trans_probs` --
    so this was already reproducible and stays so.

    It compares hash seeds, so it would have passed before the fix as well; it is a regression
    guard on the unaffected path.
    """
    script = _SEARCH_SCRIPT.format(
        graph_edges="[(1, 2), (2, 3)]", n_emitter=1, n_photon=3, seed=5
    )

    winners = {seed: _run_with_hashseed(script, seed) for seed in _SEARCH_HASH_SEEDS}

    assert len(set(winners.values())) == 1, (
        f"n_emitter=1 search gave {len(set(winners.values()))} distinct results across hash "
        f"seeds {list(winners)}; this path reaches neither changed helper, so it should not move"
    )
