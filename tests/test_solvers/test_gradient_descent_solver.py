import pytest
import matplotlib.pyplot as plt
import jax
import optax
import jax.numpy as np

import src

# src.DENSITY_MATRIX_ARRAY_LIBRARY = "jax"

from tests.test_flags import visualization, jax_library, VISUAL_TEST, JAX_TEST

import benchmarks.circuits

from src.solvers.gradient_descent_solver import GradientDescentSolver, adagrad
from src.backends.density_matrix.compiler import DensityMatrixCompiler
from src.metrics import Infidelity

from src.visualizers.density_matrix import density_matrix_bars

# todo, split into different test functions
# 1) just compute loss function and associated gradient of parameters
# 2) run few steps with gradient based solver
# 3) compute loss function + grad with different fmaps
# 4) test switching between numpy and jax - see where issues come up


def compute_loss(params, circuit, compiler, metric):
    circuit.parameters = params
    output_state = compiler.compile(circuit)
    loss = metric.evaluate(output_state, circuit)
    return loss


def run(nqubit):
    circuit, target = benchmarks.circuits.variational_entangling_layer_nqubit(nqubit)
    compiler = DensityMatrixCompiler()
    metric = Infidelity(target=target)
    return circuit, compiler, metric


def visualize(circuit, compiler, solver):
    # output_state = compiler.compile(circuit)
    # output_state.dm.draw()

    fig, ax = plt.subplots(1, 1)
    ax.plot(solver.cost_curve)
    ax.set(xlabel="Optimization Step", ylabel="Infidelity")
    plt.show()


@jax_library
def test_loss_function():
    circuit, compiler, metric = run(3)
    params = circuit.initialize_parameters()
    compute_loss(params, circuit, compiler, metric)

    loss = compute_loss(params, circuit, compiler, metric)
    grads = jax.grad(compute_loss)(params, circuit, compiler, metric)

    print(loss, grads)


@jax_library
@visualization
def test_one_layer_variational_circuit_visualize():
    circuit, compiler, metric = run(3)
    circuit.draw_circuit()


@jax_library
def test_one_layer_variational_circuit():
    circuit, compiler, metric = run(3)
    params = circuit.initialize_parameters()

    optimizer = adagrad(learning_rate=0.5)

    solver = GradientDescentSolver(metric, compiler, circuit, optimizer=optimizer)
    loss_curve, params = solver.solve(initial_params=params)

    if VISUAL_TEST:
        visualize(circuit, compiler, solver)


@jax_library
@pytest.mark.parametrize("n_qubits", [2, 3, 4])
def test_one_layer_shared_weights(n_qubits):
    circuit, compiler, metric = run(n_qubits)

    fmap = lambda: {
        id(op): op.__class__.__name__ for op in circuit.sequence(unwrapped=True)
    }
    circuit.fmap = fmap

    params = circuit.initialize_parameters()
    optimizer = adagrad(learning_rate=0.5)

    solver = GradientDescentSolver(metric, compiler, circuit, optimizer=optimizer)
    loss_curve, params = solver.solve(initial_params=params)

    if VISUAL_TEST:
        visualize(circuit, compiler, solver)


if __name__ == "__main__":
    test_one_layer_variational_circuit()
