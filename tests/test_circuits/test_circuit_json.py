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
Round-trip coverage for CircuitDAG.to_json / from_json: the generic gate arms, and the
OneQubitGateWrapper arms that are only reached when a circuit actually carries a wrapped gate.
"""

import json

import pytest

from graphiq.benchmarks.circuits import (
    ghz3_state_circuit,
    linear_cluster_3qubit_circuit,
)
from graphiq.circuit.circuit_dag import CircuitDAG
from graphiq.circuit.ops import Hadamard, OneQubitGateWrapper, SigmaX
from graphiq.utils.circuit_comparison import compare_circuits


@pytest.mark.parametrize(
    "make_circuit",
    [ghz3_state_circuit, linear_cluster_3qubit_circuit],
    ids=["ghz3", "linear_cluster_3qubit"],
)
def test_to_json_from_json_round_trips_a_measurement_circuit(make_circuit):
    # Both fixtures carry a MeasurementCNOTandReset, exercising the generic gate arms on
    # both sides of the round trip.
    circuit, _ = make_circuit()

    assert compare_circuits(circuit, CircuitDAG.from_json(circuit.to_json()))


@pytest.mark.parametrize(
    "make_circuit",
    [ghz3_state_circuit, linear_cluster_3qubit_circuit],
    ids=["ghz3", "linear_cluster_3qubit"],
)
def test_to_json_from_json_round_trips_through_a_real_json_dump(make_circuit):
    # to_json returns a dict, so the in-memory round trip above never serializes it. Pushing the
    # dict through json.dumps/json.loads turns every tuple (q_registers, q_registers_type,
    # c_registers) into a list, which must still come back as a tuple on the way in.
    circuit, _ = make_circuit()

    serialized = json.loads(json.dumps(circuit.to_json()))

    assert compare_circuits(circuit, CircuitDAG.from_json(serialized))


def test_from_json_gives_tuple_typed_registers_after_a_real_json_dump():
    # Confirms the register fields are coerced back to tuples, not merely that comparison happens
    # to still succeed on lists.
    circuit = CircuitDAG(n_emitter=1, n_photon=0, n_classical=0)
    circuit.add(Hadamard(register=0, reg_type="e"))

    serialized = json.loads(json.dumps(circuit.to_json()))
    rebuilt = CircuitDAG.from_json(serialized)

    gate = next(op for op in rebuilt.sequence() if type(op).__name__ == "Hadamard")
    assert isinstance(gate.q_registers, tuple)
    assert isinstance(gate.q_registers_type, tuple)
    assert isinstance(gate.c_registers, tuple)


def test_to_json_from_json_round_trips_a_one_qubit_gate_wrapper():
    # OneQubitGateWrapper has its own branch in to_json/from_json;
    # a plain gate circuit never reaches it.
    circuit = CircuitDAG(n_emitter=1, n_photon=0, n_classical=0)
    circuit.add(OneQubitGateWrapper([Hadamard, SigmaX], register=0, reg_type="e"))

    assert compare_circuits(circuit, CircuitDAG.from_json(circuit.to_json()))
