import pytest as pytest

from graphiq.benchmarks.graph_states import linear_cluster_state
from graphiq.solvers.alternate_target_solver import *

linear_cluster = nx.from_numpy_array(nx.to_numpy_array(linear_cluster_state(4).data))


# setting


def my_setting():
    solver_setting = AlternateGraphSolverSetting()
    # solver_setting options
    solver_setting.allow_relabel = True
    solver_setting.n_iso_graphs = 1
    solver_setting.rel_inc_thresh = 0.2
    solver_setting.allow_exhaustive = True
    solver_setting.iso_thresh = None
    solver_setting.n_lc_graphs = 1
    solver_setting.lc_orbit_depth = None
    solver_setting.lc_method = None
    solver_setting.depolarizing_rate = 0.005
    solver_setting.verbose = False
    solver_setting.save_openqasm = ""
    return solver_setting


# solver


# without noise
@pytest.mark.parametrize("target_graph", [linear_cluster, repeater_graph_states(3)])
def test_solver_no_noise(target_graph, setting=my_setting()):
    solver = AlternateTargetSolver(
        target=target_graph, solver_setting=setting, noise_model_mapping="depolarizing"
    )
    # solver options
    solver.metric = Infidelity
    solver.compiler = StabilizerCompiler()
    solver.noise_compiler = DensityMatrixCompiler()

    solver.io = None
    solver.noise_simulation = False
    solver.seed = 1
    solver.solve()


# with noise: Monte Carlo
@pytest.mark.parametrize("target_graph", [linear_cluster, repeater_graph_states(3)])
def test_solver_monte_carlo(target_graph, setting=my_setting()):
    setting.monte_carlo_params = {
        "n_sample": 20,
        "map": mcn.McNoiseMap(),
        "compiler": StabilizerCompiler(),
        "seed": 99,
        "n_parallel": 2,
        "n_single": 10,
    }
    setting.monte_carlo = True
    solver = AlternateTargetSolver(
        target=target_graph, solver_setting=setting, noise_model_mapping="depolarizing"
    )
    # solver options
    solver.metric = Infidelity
    solver.compiler = StabilizerCompiler()
    solver.noise_compiler = DensityMatrixCompiler()

    solver.io = None
    solver.noise_simulation = True
    solver.seed = 1
    solver.solve()


# with noise: density matrix noise
def test_solver_density_noise(setting=my_setting(), target_graph=linear_cluster):
    setting.monte_carlo = False
    solver = AlternateTargetSolver(
        target=target_graph, solver_setting=setting, noise_model_mapping="depolarizing"
    )
    # solver options
    solver.metric = Infidelity
    solver.compiler = StabilizerCompiler()
    solver.noise_compiler = DensityMatrixCompiler()

    solver.io = None
    solver.noise_simulation = True
    solver.seed = 1
    solver.solve()
