from src.solvers.rule_based_random_solver import RuleBasedRandomSearchSolver
import matplotlib.pyplot as plt
from src.backends.density_matrix.compiler import DensityMatrixCompiler
from src.circuit import CircuitDAG
from src.metrics import MetricFidelity
import src.backends.density_matrix.functions as dmf
from src.visualizers.density_matrix import density_matrix_bars

from benchmarks.circuits import *




def test_solver_initialization():
    circuit_ideal, state_ideal = linear_cluster_4qubit_circuit()
    target = state_ideal['dm']

    compiler = DensityMatrixCompiler()
    metric = MetricFidelity(target=target)

    solver = RuleBasedRandomSearchSolver(target=target, metric=metric, compiler=compiler, n_emitter=1, n_photon=3)
    solver.test_initialization(10)


def test_solver_3qubit():
    circuit_ideal, state_ideal = linear_cluster_3qubit_circuit()
    target = state_ideal['dm']
    n_photon = 3
    n_emitter = 1
    compiler = DensityMatrixCompiler()
    metric = MetricFidelity(target=target)

    solver = RuleBasedRandomSearchSolver(target=target, metric=metric, compiler=compiler, n_emitter=n_emitter, n_photon=n_photon)
    solver.solve(10)
    print(solver.hof[0][0])

    state = compiler.compile(solver.hof[1][0])
    circuit = solver.hof[1][0]
    circuit.draw_circuit()
    # circuit2 = solver.hof[1][5]
    # circuit2.draw_circuit()
    # circuit.draw_dag()
    fig, axs = density_matrix_bars(target)
    fig.suptitle("TARGET DENSITY MATRIX")
    plt.show()

    state = dmf.partial_trace(state.data, list(range(n_photon)), (n_photon + n_emitter) * [2])

    print(metric.evaluate(state, circuit))
    fig, axs = density_matrix_bars(state)

    fig.suptitle("CREATED DENSITY MATRIX")
    plt.show()


def test_solver_4qubit():
    circuit_ideal, state_ideal = linear_cluster_4qubit_circuit()
    target = state_ideal['dm']
    n_photon = 4
    n_emitter = 1
    compiler = DensityMatrixCompiler()
    metric = MetricFidelity(target=target)

    solver = RuleBasedRandomSearchSolver(target=target, metric=metric, compiler=compiler, n_emitter=n_emitter, n_photon=n_photon)


    solver.solve(20)
    print('hof score is '+str(solver.hof[0][0]))
    circuit = solver.hof[1][0]
    state = compiler.compile(circuit)
    state2 = compiler.compile(circuit)
    state3 = compiler.compile(circuit)
    circuit.draw_circuit()
    # circuit.draw_dag()
    fig, axs = density_matrix_bars(target)
    fig.suptitle("TARGET DENSITY MATRIX")
    plt.show()

    new_state = dmf.partial_trace(state.data, keep=list(range(n_photon)), dims=(n_photon + n_emitter) * [2])
    new_state2 = dmf.partial_trace(state2.data, keep=list(range(n_photon)), dims=(n_photon + n_emitter) * [2])
    new_state3 = dmf.partial_trace(state3.data, keep=list(range(n_photon)), dims=(n_photon + n_emitter) * [2])
    print('Are these two states the same: '+str(np.allclose(new_state, new_state3)))
    print('The circuit compiles a state that has an infidelity '+ str(metric.evaluate(new_state, circuit)))
    fig, axs = density_matrix_bars(new_state)

    fig.suptitle("CREATED DENSITY MATRIX")
    plt.show()



def test_solver_GHZ3qubit():
    circuit_ideal, state_ideal = ghz3_state_circuit()
    target = state_ideal['dm']
    n_photon = 3
    n_emitter = 1
    compiler = DensityMatrixCompiler()
    metric = MetricFidelity(target=target)

    solver = RuleBasedRandomSearchSolver(target=target, metric=metric, compiler=compiler, n_emitter=n_emitter, n_photon=n_photon)
    solver.solve(100)
    print(solver.hof[0][0])

    state = compiler.compile(solver.hof[1][0])
    circuit = solver.hof[1][0]
    circuit.draw_circuit()

    fig, axs = density_matrix_bars(target)
    fig.suptitle("TARGET DENSITY MATRIX")
    plt.show()

    state = dmf.partial_trace(state.data, list(range(n_photon)), (n_photon + n_emitter) * [2])

    print(metric.evaluate(state, circuit))
    fig, axs = density_matrix_bars(state)

    fig.suptitle("CREATED DENSITY MATRIX")
    plt.show()