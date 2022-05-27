"""
Classes to compute metrics on a circuit and/or system states
"""

from abc import ABC, abstractmethod
import numpy as np


class MetricBase(ABC):
    """
    Base class for a metric.

    Metrics should be scalar values computed on the circuit and/or system states
    If a metric is used as a cost function, we aim to minimize it (i.e. smaller metric means better performance); this
    is not, however, required of metrics in general.
    """

    def __init__(self, log_steps=1, *args, **kwargs):
        """
        Create a MetricBase object

        :param log_steps: the metric values are computed at every log_steps optimization step
        :type log_steps: int
        :return: the function returns nothing
        :rtype: None
        """
        self.log_steps = log_steps
        self.name = "base"
        self.differentiable = False

        self.log = []  # will store the metric evaluations
        self._inc = 0  #

    @abstractmethod
    def evaluate(self, state, circuit):
        raise NotImplementedError("Please use an inherited class, not the base metric class")

    def increment(self):
        """
        Counts up the number of times a given metric has been evaluated

        :return: this function returns nothing
        :rtype: None
        """
        self._inc += 1


class MetricFidelity(MetricBase):
    def __init__(self, ideal_state, log_steps=1, *args, **kwargs):
        """
        Creates a Fidelity Metric object (which computes fidelity with respect to the ideal_state

        :param ideal_state: the ideal state against which we compute fidelity
        :type ideal_state: QuantumState
        :param log_steps: the metric values are computed at every log_steps optimization step
        :type log_steps: int
        :return: the function returns nothing
        :rtype: None
        """
        super().__init__(log_steps=log_steps, *args, **kwargs)
        self.ideal_state = ideal_state
        self.differentiable = False

    def evaluate(self, state, circuit):
        """
        Evaluates the fidelity from a given state and circuit

        :param state: the state to evaluate
        :type state: QuantumState
        :param circuit: circuit which generated state
                        Not used for the fidelity evaluation, but argument is provided for API consistency
        :type circuit: CircuitBase (or subclass of it)
        :return: the fidelity
        :rtype: float
        """
        # TODO: replace by actual fidelity check
        val = np.random.random()
        self.increment()

        if self._inc % self.log_steps == 0:
            self.log.append(val)

        return val


class MetricCircuitDepth(MetricBase):
    """
    A Circuit Depth based metric object
    """
    def __init__(self, log_steps=1, depth_penalty=None, *args, **kwargs):
        """
        Create MetricCircuitDepth object

        :param log_steps: the metric values are computed at every log_steps optimization step
        :type log_steps: int
        :param depth_penalty: a function which calculates a "cost"/penalty as a function of circuit depth
        :type depth_penalty: function
        :return: the function returns nothing
        :rtype: None
        """
        super().__init__(log_steps=log_steps, *args, **kwargs)
        self.differentiable = False
        if depth_penalty is None:
            self.depth_penalty = lambda x: x  # by default, the penalty for depth is the depth itself
        else:
            self.depth_penalty = depth_penalty

    def evaluate(self, state, circuit):
        """
        Calculates a scalar function of the circuit depth

        :param state: state which was created by the circuit. 
        :type state:
        :param circuit:
        :type circuit:
        :return: the scalar penalty resulting from circuit depth. By default, this is the circuit depth itself
        :rtype: float or int
        """
        # TODO: replace this by an actual circuit depth evaluation
        depth = np.random.random()
        val = self.depth_penalty(depth)
        self.increment()

        if self._inc % self.log_steps == 0:
            self.log.append(val)

        return val


class Metrics(object):
    """
    Wraps around one or more metric functions, evaluating each and logging the values
    """
    _all = {  # metrics that can be used, which can be specified by the dictionary keys or as a class instance
        "fidelity": MetricFidelity,
        "circuit-depth": MetricCircuitDepth,
    }

    def __init__(self, metrics: list):

        # pass in either list of either strings or of specific Metric instance (must be an accepted Metric)
        _metrics = []
        for metric in metrics:
            if metric.__class__ in self._all.values():
                _metrics.append(metric)
            elif metric in self._all.keys():
                _metrics.append(self._all[metric]())
            else:
                raise UserWarning(f"{metric} is not a recognized metric - it will not be evaluated.")
        self._metrics = _metrics

    def evaluate(self, state, circuit):
        for i, metric in enumerate(self._metrics):
            res = metric.evaluate(state, circuit)

    @property
    def log(self):
        m = {}
        for i, metric in enumerate(self._metrics):
            m[metric.__class__.__name__] = metric.log
        return m


if __name__ == "__main__":

    # set how often to log the metric evaluations
    MetricBase.log_steps = 3
    MetricFidelity.log_steps = 1

    metrics = Metrics([
        MetricCircuitDepth(),
        MetricFidelity()
    ])

    for _ in range(10):
        metrics.evaluate(state=None, circuit=None)

    print(metrics.log)
