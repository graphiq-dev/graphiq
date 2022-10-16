"""
Gradient descent algorithms on parameterized circuits.
"""
import tqdm

import src
from src.backends.compiler_base import CompilerBase
from src.solvers import SolverBase
from src.circuit import CircuitBase, CircuitDAG
from src.metrics import MetricBase

from src.io import IO

import optax
import jax

# standard optimizers
from optax import adam, adagrad, rmsprop, adamw, adabelief


class GradientDescentSolver(SolverBase):
    """
    A solver class based on gradient descent algorithms.
    Requires the use of `jax` as the numerical backend.
    The optimizer routine is provided by the `optax` package.
    """

    name = "gradient-descent-solver"

    def __init__(
        self,
        metric: MetricBase,
        compiler: CompilerBase,
        circuit: CircuitDAG = None,
        io: IO = None,
        optimizer=None,
        n_step=30,
        progress=True,
        *args,
        **kwargs,
    ):
        """ """
        super().__init__(
            None,
            metric,
            compiler,
            circuit,
            io,
        )
        if src.DENSITY_MATRIX_ARRAY_LIBRARY != "jax":
            raise RuntimeError("JAX not being used as array backend.")

        self.optimizer = optimizer
        self.n_step = n_step
        self.progress = progress
        self.loss_curve = []

    @staticmethod
    def compute_loss(
        params: dict, circuit: CircuitBase, compiler: CompilerBase, metric: MetricBase
    ):
        """
        Wrapper for simulating and evaluating parameter values.

        :param params: parameter dictionary
        :type params: dict
        :param circuit: parameterized circuit object
        :type circuit: CircuitBase
        :param compiler: density matrix compiler
        :param metric: metric to evaluate performance of circuit parameters
        :return:
        """
        # sets parameter, simulates circuit, evaluates and then returns the loss
        circuit.parameters = params
        output_state = compiler.compile(circuit)
        loss = metric.evaluate(output_state, circuit)
        return loss

    def solve(self, initial_params=None):
        """
        Main gradient descent algorithm, performing `n_steps` updates with the provided optimizer.
        :param initial_params: initial parameter dictionary
        :type initial_params: dict
        :return:
        """
        if initial_params is not None:
            params = initial_params
        else:
            params = self.circuit.initialize_parameters()

        opt_state = self.optimizer.init(params)

        loss_curve = []
        for step in (
            pbar := tqdm.tqdm(range(self.n_step), disable=(not self.progress))
        ):
            loss = self.compute_loss(params, self.circuit, self.compiler, self.metric)
            grads = jax.grad(self.compute_loss)(
                params, self.circuit, self.compiler, self.metric
            )
            updates, opt_state = self.optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            loss_curve.append(loss)
            if self.progress:
                pbar.set_description(f"Cost: {loss:.10f}")
            else:
                print(step, loss, params)

        self.loss_curve = loss_curve
        return loss_curve, params
