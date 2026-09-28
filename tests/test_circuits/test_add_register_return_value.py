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
from graphiq.circuit.circuit_dag import CircuitDAG


def test_add_emitter_register_returns_index():
    dag = CircuitDAG(n_emitter=0, n_photon=0, n_classical=0)

    assert dag.add_emitter_register() == 0
    assert dag.add_emitter_register() == 1
    assert dag.add_emitter_register() == 2


def test_add_photonic_and_classical_registers_number_independently():
    dag = CircuitDAG(n_emitter=0, n_photon=0, n_classical=0)

    assert dag.add_emitter_register() == 0
    assert dag.add_emitter_register() == 1

    assert dag.add_photonic_register() == 0
    assert dag.add_photonic_register() == 1
    assert dag.add_photonic_register() == 2

    assert dag.add_classical_register() == 0
    assert dag.add_classical_register() == 1

    # Adding another emitter register continues the emitter numbering,
    # unaffected by the photonic/classical registers added above.
    assert dag.add_emitter_register() == 2
