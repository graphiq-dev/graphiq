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
# import jax
# import jax.numpy as np
# import matplotlib.pyplot as plt
# import pytest
# from optax import adagrad
#
# from graphiq.backends.density_matrix.compiler import DensityMatrixCompiler
# from graphiq.metrics import Infidelity
# from graphiq.solvers.gradient_descent_solver import GradientDescentSolver
# from tests.test_flags import visualization, jax_library, VISUAL_TEST
# from graphiq.benchmarks.circuits import strongly_entangling_layer
#
# import graphiq
# graphiq.DENSITY_MATRIX_ARRAY_LIBRARY = "jax"
#
#
# # todo, split into different test functions
# # 1) just compute loss function and associated gradient of parameters (done)
# # 2) run few steps with gradient based solver (done)
# # 3) compute loss function + grad with different fmaps (done)
# # 4) test switching between numpy and jax - see where issues come up
#
#
# def compute_loss(params, circuit, compiler, metric):
#     circuit.parameters = params
#     output_state = compiler.compile(circuit)
#     loss = metric.evaluate(output_state, circuit)
#     return loss
#
#
# def run(nqubit):
#     circuit, target = strongly_entangling_layer(nqubit)
#     compiler = DensityMatrixCompiler()
#     metric = Infidelity(target=target)
#     return circuit, compiler, metric
#
#
# def visualize(circuit, compiler, solver):
#     fig, ax = plt.subplots(1, 1)
#     ax.plot(solver.cost_curve)
#     ax.set(xlabel="Optimization Step", ylabel="Infidelity")
#     plt.show()
#
#
# @jax_library
# def test_loss_function():
#     circuit, compiler, metric = run(3)
#     params = circuit.initialize_parameters()
#     compute_loss(params, circuit, compiler, metric)
#
#     loss = compute_loss(params, circuit, compiler, metric)
#     grads = jax.grad(compute_loss)(params, circuit, compiler, metric)
#
#     print(loss, grads)
#     assert loss is not np.nan
#
#
# @jax_library
# @visualization
# def test_one_layer_variational_circuit_visualize():
#     circuit, compiler, metric = run(3)
#     circuit.draw_circuit()
#
#
# @jax_library
# def test_one_layer_variational_circuit():
#     circuit, compiler, metric = run(2)
#     params = circuit.initialize_parameters(seed=0)
#
#     optimizer = adagrad(learning_rate=0.25)
#
#     solver = GradientDescentSolver(
#         metric, compiler, circuit, optimizer=optimizer, n_step=150
#     )
#     loss_curve, params = solver.solve(initial_params=params)
#
#     assert np.isclose(loss_curve[-1], 0.0)  # check that the solver converged
#
#     if VISUAL_TEST:
#         visualize(circuit, compiler, solver)
#
#
# @jax_library
# @pytest.mark.parametrize("n_qubits", [2, 3, 4])
# def test_one_layer_shared_weights(n_qubits):
#     circuit, compiler, metric = run(n_qubits)
#
#     fmap = lambda: {
#         id(op): op.__class__.__name__ for op in circuit.sequence(unwrapped=True)
#     }
#     circuit.fmap = fmap
#
#     params = circuit.initialize_parameters()
#     optimizer = adagrad(learning_rate=0.5)
#
#     solver = GradientDescentSolver(
#         metric, compiler, circuit, optimizer=optimizer, n_step=30
#     )
#     loss_curve, params = solver.solve(initial_params=params)
#
#     if VISUAL_TEST:
#         visualize(circuit, compiler, solver)
#
#
# if __name__ == "__main__":
#     test_one_layer_variational_circuit()
