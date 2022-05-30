"""
Classes to compute metrics on a circuit and/or system states
"""

from abc import ABC, abstractmethod
import numpy as np

from benchmarks.circuits import bell_state_circuit


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

        :param state: state which was created by the circuit. This is not actually used by this metric object,
                      but is nonetheless provided to guarantee a uniform API between Metric-type objects
        :type state: QuantumState
        :param circuit: the circuit to evaluate
        :type circuit: CircuitBase (or a subclass of it)
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

    def __init__(self, metrics_list: list):
        """
        Create a Metrics object which acts as a wrapper around Metric functions

        :param metrics_list: metrics to evaluate
        :type metrics_list: list of strings (strings should be metric names)
        :return: function returns nothing
        :rtype: None
        """
        # pass in either list of either strings or of specific Metric instance (must be an accepted Metric)
        _metrics = []
        for metric in metrics_list:
            if metric.__class__ in self._all.values():
                _metrics.append(metric)
            elif metric in self._all.keys():
                _metrics.append(self._all[metric]())
            else:
                raise UserWarning(f"{metric} is not a recognized metric - it will not be evaluated.")
        self._metrics = _metrics

    def evaluate(self, state, circuit):
        """
        Evaluate each metric function contained by the Metrics object

        :param state: the state on which to evaluate the metrics
        :type state: QuantumState
        :param circuit: the circuit on which to evaluate the metrics
        :type circuit: CircuitBase (or a subclass of it)
        :return: this function returns nothing
        :rtype: None
        """
        for i, metric in enumerate(self._metrics):
            metric.evaluate(state, circuit)

    @property
    def log(self):
        """
        The joint log of all metric functions

        :return: the log itself
        :rtype: dict (keys are metric class names, values are the logs)
        """
        m = {}
        for i, metric in enumerate(self._metrics):
            # TODO: switch the key to the strings provided in __init__ (abstracts things better from the user)
            m[metric.__class__.__name__] = metric.log
        return m


if __name__ == "__main__":
    """ Metric usage example """
    # set how often to log the metric evaluations
    MetricBase.log_steps = 3
    MetricFidelity.log_steps = 1

    _, ideal_state = bell_state_circuit()

    metrics = Metrics([
        MetricCircuitDepth(),
        MetricFidelity(ideal_state)
    ])

    for _ in range(10):
        metrics.evaluate(state=None, circuit=None)

    print(metrics.log)
