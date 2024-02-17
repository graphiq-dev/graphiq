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
from graphiq.backends.stabilizer.state import *


def test_state1():
    state = Stabilizer(4)
    print(state.tableau.destabilizer_to_labels())
    print(state.tableau.stabilizer_to_labels())
    state.tableau.stabilizer_z = np.array(
        [[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]]
    )
    state.tableau.destabilizer_z = np.array(
        [[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]]
    )
    state.apply_hadamard(1)
    state.tableau.destabilizer_from_labels(["XYXI", "ZYZI", "YIII", "ZZZZ"])
    print(state.tableau.destabilizer)
