import pytest as pytest
import networkx as nx
from src.backends.density_matrix.compiler import DensityMatrixCompiler
from src.backends.stabilizer.compiler import StabilizerCompiler
from src.metrics import Infidelity
import src.noise.monte_carlo_noise as mcn
from src.solvers.hybrid_solvers import HybridGraphSearchSolverSetting
from src.solvers.hybrid_solvers import AlternateGraphSolver

from benchmarks.graph_states import repeater_graph_states, linear_cluster_state
linear_cluster = nx.from_numpy_array(nx.to_numpy_array(linear_cluster_state(4).data))
# setting
solver_setting = HybridGraphSearchSolverSetting()
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


# solver

# without noise
@pytest.mark.parametrize("target_graph", [linear_cluster, repeater_graph_states(3)])
def test_solver_no_noise(target_graph, setting=solver_setting):
    solver = AlternateGraphSolver(target_graph=target_graph,
                                  graph_solver_setting=setting,
                                  noise_model_mapping="depolarizing")
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
def test_solver_monte_carlo(target_graph, setting=solver_setting):
    setting.monte_carlo_params = {
        "n_sample": 20,
        "map": mcn.McNoiseMap(),
        "compiler": StabilizerCompiler(),
        "seed": 99,
        "n_parallel": 2,
        "n_single": 10,
    }
    setting.monte_carlo = True
    solver = AlternateGraphSolver(target_graph=target_graph,
                                  graph_solver_setting=setting,
                                  noise_model_mapping="depolarizing")
    # solver options
    solver.metric = Infidelity
    solver.compiler = StabilizerCompiler()
    solver.noise_compiler = DensityMatrixCompiler()

    solver.io = None
    solver.noise_simulation = True
    solver.seed = 1
    solver.solve()


# with noise: density matrix noise
def test_solver_density_noise(setting=solver_setting, target_graph=linear_cluster):
    setting.monte_carlo = False
    solver = AlternateGraphSolver(target_graph=target_graph,
                                  graph_solver_setting=setting,
                                  noise_model_mapping="depolarizing")
    # solver options
    solver.metric = Infidelity
    solver.compiler = StabilizerCompiler()
    solver.noise_compiler = DensityMatrixCompiler()

    solver.io = None
    solver.noise_simulation = True
    solver.seed = 1
    solver.solve()
