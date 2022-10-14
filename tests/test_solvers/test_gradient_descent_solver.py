import src

src.DENSITY_MATRIX_ARRAY_LIBRARY = "jax"

import matplotlib.pyplot as plt
import optax
import jax

from src import ops
from src.circuit import CircuitDAG
from src.solvers.gradient_descent_solver import GradientDescentSolver, adagrad
import src.backends.density_matrix.functions as dmf
from src.backends.density_matrix.compiler import DensityMatrixCompiler
from src.backends.density_matrix.state import DensityMatrix
from src.state import QuantumState
from src.metrics import Infidelity

import benchmarks.circuits


_, target = benchmarks.circuits.bell_state_circuit()
# _, target = benchmarks.circuits.ghz3_state_circuit()
# _, target = benchmarks.circuits.ghz4_state_circuit()

compiler = DensityMatrixCompiler()
metric = Infidelity(target=target)

target.dm.draw()


circuit = CircuitDAG(n_emitter=target.n_qubits, n_photon=0, n_classical=0)

circuit.add(ops.ParameterizedOneQubitRotation(register=0, reg_type="e"))
circuit.add(
    ops.ParameterizedControlledRotationQubit(
        control=0, control_type="e", target=1, target_type="e"
    )
)
# circuit.add(ops.ParameterizedControlledRotationQubit(control=1, control_type="e", target=2, target_type="e"))
# circuit.add(ops.ParameterizedControlledRotationQubit(control=2, control_type="e", target=3, target_type="e"))


circuit = CircuitDAG(n_emitter=target.n_qubits, n_photon=0, n_classical=0)
circuit.add(
    ops.ParameterizedOneQubitRotation(
        register=0,
        reg_type="e",
        params=(0.0, 1.0, 1.0),  # set the parameters explicitly, if desired
    )
)
for i in range(1, target.n_qubits):
    circuit.add(
        ops.ParameterizedControlledRotationQubit(
            control=i - 1, control_type="e", target=i, target_type="e"
        )
    )


circuit.draw_circuit()

# fmap = lambda: {id(op): op.__class__.__name__ for op in circuit.sequence(unwrapped=True)}
# circuit.fmap = fmap
params = circuit.initialize_parameters()  # randomly sample the initial parameters
print(circuit.parameters)

# todo, split into different test functions
# 1) just compute loss function and associated gradient of parameters
# 2) run few steps with gradient based solver
# 3) compute loss function + grad with different fmaps
# 4) test switching between numpy and jax - see where issues come up


def compute_loss(params: dict, circuit: CircuitDAG):
    circuit.parameters = params
    output_state = compiler.compile(circuit)
    loss = metric.evaluate(output_state, circuit)
    return loss


loss = compute_loss(params, circuit)
print(loss)

optimizer = adagrad(learning_rate=0.5)
opt_state = optimizer.init(circuit.parameters)

solver = GradientDescentSolver(target, metric, compiler, circuit, optimizer=optimizer)
loss_curve, params = solver.solve(initial_params=params)

# loss_curve = []
# for i in range(50):
#     loss = compute_loss(params, circuit)
#     grads = jax.grad(compute_loss)(params, circuit)
#     updates, opt_state = optimizer.update(grads, opt_state)
#     params = optax.apply_updates(params, updates)
#     loss_curve.append(loss)
#     print(i, loss, params)

circuit.parameters = params
output_state = compiler.compile(circuit)
output_state.dm.draw()

fig, ax = plt.subplots(1, 1)
ax.plot(loss_curve)
ax.set(xlabel="Optimization Step", ylabel="Infidelity")
plt.show()
