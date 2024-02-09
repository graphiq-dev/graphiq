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
