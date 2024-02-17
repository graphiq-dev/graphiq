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
from graphiq.solvers.solver_base import *


# Test for RandomSearchSolverSetting class
def test_random_search_solver_setting_init():
    solver_setting = RandomSearchSolverSetting()

    assert solver_setting.n_hof == 5
    assert solver_setting.n_pop == 50
    assert solver_setting.n_stop == 50


def test_random_search_solver_setting_getter_setter():
    solver_setting = RandomSearchSolverSetting()

    solver_setting.n_hof = 10
    solver_setting.n_pop = 10
    solver_setting.n_stop = 10

    assert solver_setting.n_hof == 10
    assert solver_setting.n_pop == 10
    assert solver_setting.n_stop == 10
