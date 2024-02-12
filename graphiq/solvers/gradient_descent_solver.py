"""
Gradient descent algorithms on parameterized circuits.
"""

import jax
import optax
import tqdm

import graphiq
from graphiq.backends.compiler_base import CompilerBase
from graphiq.circuit.circuit_base import CircuitBase
from graphiq.circuit.circuit_dag import CircuitDAG
from graphiq.io import IO
from graphiq.metrics import MetricBase
from graphiq.solvers import SolverBase


# standard optimizers


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
        if graphiq.DENSITY_MATRIX_ARRAY_LIBRARY != "jax":
            raise RuntimeError("JAX not being used as array backend.")

        self.optimizer = optimizer
        self.n_step = n_step
        self.progress = progress
        self.cost_curve = []

    @staticmethod
    def compute_cost(
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
        :return: cost: scalar value quantifying the circuit performance with the given parameter set
        """
        # sets parameter, simulates circuit, evaluates and then returns the cost
        circuit.parameters = params
        output_state = compiler.compile(circuit)
        cost = metric.evaluate(output_state, circuit)
        return cost

    def solve(self, initial_params=None):
        """
        Main gradient descent algorithm, performing `n_steps` updates with the provided optimizer.

        :param initial_params: initial parameter dictionary
        :type initial_params: dict
        :return: cost_curve, params: list of computed cost values at each optimization step, and final optimized params
        :rtype: (list, dict)
        """
        if initial_params is not None:
            params = initial_params
        else:
            params = self.circuit.initialize_parameters()

        opt_state = self.optimizer.init(params)

        cost_curve = []
        grad = jax.grad(self.compute_cost)
        for step in (
            pbar := tqdm.tqdm(range(self.n_step), disable=(not self.progress))
        ):
            cost = self.compute_cost(params, self.circuit, self.compiler, self.metric)
            gradient = grad(params, self.circuit, self.compiler, self.metric)
            updates, opt_state = self.optimizer.update(gradient, opt_state)
            params = optax.apply_updates(params, updates)
            cost_curve.append(cost)
            if self.progress:
                pbar.set_description(f"Cost: {cost:.10f}")
            else:
                print(step, cost, params)

        self.cost_curve = cost_curve
        return cost_curve, params
