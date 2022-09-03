"""

"""
import tqdm

from src.backends.compiler_base import CompilerBase
from src.solvers import SolverBase
from src.circuit import CircuitBase, CircuitDAG
from src.metrics import MetricBase

from src.io import IO

import optax
import jax


class GradientDescentSolver(SolverBase):
    """ """

    name = "gradient-descent-solver"

    def __init__(
        self,
        target,
        metric: MetricBase,
        compiler: CompilerBase,
        circuit: CircuitDAG = None,
        io: IO = None,
        optimizer=None,
        n_step=300,
        progress=True,
        *args,
        **kwargs,
    ):
        """ """
        super().__init__(
            target,
            metric,
            compiler,
            circuit,
            io,
        )

        self.optimizer = optimizer
        self.n_step = n_step
        self.progress = progress

    @staticmethod
    def compute_loss(
        params: dict, circuit: CircuitBase, compiler: CompilerBase, metric: MetricBase
    ):
        # sets parameter, compiles and simulates circuit, evaluates and then returns the loss
        circuit.set_params(params)
        output_state = compiler.compile(circuit)
        loss = metric.evaluate(output_state, circuit)
        return loss

    def solve(self):
        # updates the `params` iteratively for `n_steps`
        params = self.circuit.collect_params()
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
        return loss_curve, params

    @staticmethod
    def adam(
        learning_rate: float = 0.5,
        b1: float = 0.9,
        b2: float = 0.999,
        eps: float = 1e-8,
        eps_root: float = 0.0,
    ):
        return optax.adam(
            learning_rate=learning_rate, b1=b1, b2=b2, eps=eps, eps_root=eps_root
        )
