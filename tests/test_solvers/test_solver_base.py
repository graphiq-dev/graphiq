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


# Test for RandomSearchSolver.tournament_selection. The payload is (score, [score]) rather than a
# plain (float, str) pair or a CircuitDAG: a (float, str) pair would round-trip through
# copy.deepcopy as the very same object (CPython treats it as atomic), defeating the identity
# check below, and CircuitDAG has no __eq__, so the k=1 equality check would fail regardless of
# whether the copy is otherwise correct. A one-element list is mutable enough to force a real
# deep copy while still comparing equal by value.


def test_tournament_selection_k_zero_is_a_no_op():
    solver = RandomSearchSolver(None, None, None)
    population = [(0.5, [0.5]), (0.1, [0.1])]

    assert solver.tournament_selection(population, k=0) is population


def test_tournament_selection_samples_with_replacement_and_returns_deep_copies():
    # random.choices(population, k=k) samples with replacement, so k == len(population) does
    # not guarantee the minimum-scoring entry is ever drawn; only the length, copy-ness, and
    # score bound are guaranteed.
    random.seed(0)
    solver = RandomSearchSolver(None, None, None)
    population = [(0.5, [0.5]), (0.1, [0.1]), (0.9, [0.9])]
    input_scores = [score for score, _ in population]

    new_population = solver.tournament_selection(population, k=2)

    assert len(new_population) == solver.setting.n_pop
    for score, payload in new_population:
        assert score in input_scores
    assert not any(entry is p for entry in new_population for p in population)


def test_tournament_selection_k_one_with_a_single_element_population_copies_that_element():
    random.seed(0)
    solver = RandomSearchSolver(None, None, None)
    element = (0.42, [0.42])
    population = [element]

    new_population = solver.tournament_selection(population, k=1)

    assert len(new_population) == solver.setting.n_pop
    for entry in new_population:
        assert entry == element
        assert entry is not element
