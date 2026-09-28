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
run_circuit(..., reverse=True) runs the inverse circuit without changing the
caller's gate list.
"""

from graphiq.backends.stabilizer.clifford_tableau import CliffordTableau
from graphiq.backends.stabilizer.functions.transformation import run_circuit
from graphiq.backends.stabilizer.state import Stabilizer

# the X gate puts a sign in the forward state, so a round trip catches a phase error
_GATES = [("H", 0), ("P", 0), ("CNOT", 0, 1), ("X", 1)]


def test_run_circuit_reverse_leaves_the_gate_list_unchanged():
    gates = list(_GATES)
    run_circuit(CliffordTableau(2), gates, reverse=True)
    assert gates == _GATES


def test_run_circuit_reverse_twice_with_one_list_gives_the_same_state():
    gates = list(_GATES)
    first = run_circuit(CliffordTableau(2), gates, reverse=True)
    second = run_circuit(CliffordTableau(2), gates, reverse=True)
    assert first == second
    # after two uses the same list still inverts the circuit
    forward = run_circuit(CliffordTableau(2), list(_GATES))
    assert forward != CliffordTableau(2)
    assert run_circuit(forward, gates, reverse=True) == CliffordTableau(2)


def test_stabilizer_apply_circuit_reverse_leaves_the_gate_list_unchanged():
    gates = list(_GATES)
    Stabilizer(2).apply_circuit(gates, reverse=True)
    assert gates == _GATES
