"""
Classes to compute metrics on a circuit and/or system states
"""

from abc import ABC, abstractmethod
import numpy as np

import src.backends.density_matrix.functions as dmf
import src.backends.stabilizer.functions.metric as sfm


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
        raise NotImplementedError(
            "Please use an inherited class, not the base metric class"
        )

    def increment(self):
        """
        Counts up the number of times a given metric has been evaluated

        :return: this function returns nothing
        :rtype: None
        """
        self._inc += 1


class Infidelity(MetricBase):
    def __init__(self, target, log_steps=1, *args, **kwargs):
        """
        Creates an Infidelity Metric object, which computes 1-fidelity with respect to the ideal_state

        :param target: the ideal state against which we compute fidelity
        :type target: QuantumState
        :param log_steps: the metric values are computed at every log_steps optimization step
        :type log_steps: int
        :raises AssertionError: if targe is not a valid density matrix
        :return: nothing
        :rtype: None
        """
        super().__init__(log_steps=log_steps, *args, **kwargs)
        self.target = target
        self.differentiable = False

    def evaluate(self, state, circuit):
        """
        Evaluates the infidelity from a given state and circuit


        :param state: the state to evaluate
        :type state: QuantumState
        :param circuit: circuit which generated state
                        Not used for the fidelity evaluation, but argument is provided for API consistency
        :type circuit: CircuitBase (or subclass of it)
        :raises AssertionError: if the state is not a valid density matrix
        :return: infidelity = 1 - fidelity
        :rtype: float
        """
        # TODO: add check for the representation
        if state._stabilizer is not None and self.target._stabilizer is not None:
            fid = sfm.fidelity(self.target.stabilizer.data, state.stabilizer.data)
        elif state._dm is not None and self.target._dm is not None:
            fid = dmf.fidelity(self.target.dm.data, state.dm.data)
        else:
            raise ValueError("Cannot compute the infidelity.")
        self.increment()

        if self._inc % self.log_steps == 0:
            self.log.append(1 - fid)

        return 1 - fid


class TraceDistance(MetricBase):
    def __init__(self, target, log_steps=1, *args, **kwargs):
        """
        Creates a TraceDistance Metric object, which computes the trace distance between the current state and the
        target state.

        :param target: the ideal state against which we compute fidelity
        :type target: QuantumState
        :param log_steps: the metric values are computed at every log_steps optimization step
        :type log_steps: int
        :return: nothing
        :rtype: None
        """
        super().__init__(log_steps=log_steps, *args, **kwargs)
        self.target = target
        self.differentiable = False

    def evaluate(self, state, circuit):
        """
        Evaluates the trace distance from a given state and circuit

        :param state: the state to evaluate
        :type state: QuantumState
        :param circuit: circuit which generated state
                        Not used for the trace distance evaluation, but argument is provided for API consistency
        :type circuit: CircuitBase (or subclass of it)
        :return: the trace distance
        :rtype: float
        """

        trace_distance = dmf.trace_distance(self.target.dm.data, state.dm.data)
        self.increment()

        if self._inc % self.log_steps == 0:
            self.log.append(trace_distance)

        return trace_distance


class CircuitDepth(MetricBase):
    """
    A Circuit Depth based metric object
    """

    def __init__(self, log_steps=1, depth_penalty=None, *args, **kwargs):
        """
        Create CircuitDepth object

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
            self.depth_penalty = (
                lambda x: x
            )  # by default, the penalty for depth is the depth itself
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

        depth = circuit.depth
        val = self.depth_penalty(depth)
        self.increment()

        if self._inc % self.log_steps == 0:
            self.log.append(val)

        return val


class Metrics(MetricBase):
    """
    Wraps around one or more metric functions, evaluating each and logging the values
    """

    _all = {  # metrics that can be used, which can be specified by the dictionary keys or as a class instance
        "Infidelity": Infidelity,
        "TraceDistance": TraceDistance,
        "CircuitDepth": CircuitDepth,
    }

    def __init__(
        self, metrics_list: list, metric_weight=None, log_steps=1, *args, **kwargs
    ):
        """
        Create a Metrics object which acts as a wrapper around Metric functions

        :param metrics_list: metrics to evaluate
        :type metrics_list: list of strings (strings should be metric names) OR MetricBase objects
                            MetricBase objects may be preferable, since we cannot set initial parameters via the list of strings.
        :param metric_weight: some representation of how to weigh the different metric results against one another
                              if None, all metrics provided are weighted equally (by 1)
                              if a list or ndarray, the metrics are a linear combination weighted by the list/ndarray values
                              Otherwise, metric_weight is a function, that can make any mathematical function of the individual
                              metric values.
        :type metric_weight: None, numpy.ndarray, list, or Function
        :param log_steps: the metric values are computed at every log_steps optimization step
        :type log_steps: int
        :return: function returns nothing
        :rtype: None
        """
        super().__init__(log_steps=log_steps, *args, **kwargs)
        # pass in either list of either strings or of specific Metric instance (must be an accepted Metric)
        _metrics = []
        for metric in metrics_list:
            if metric.__class__ in self._all.values():
                _metrics.append(metric)
            elif metric in self._all.keys():
                _metrics.append(self._all[metric]())
            else:
                raise UserWarning(
                    f"{metric} is not a recognized metric - it will not be evaluated."
                )
        self._metrics = _metrics

        if (
            metric_weight is None
            or isinstance(metric_weight, list)
            or isinstance(metric_weight, np.ndarray)
        ):
            if metric_weight is None:
                metric_weight = np.ones(len(metrics_list)).flatten()

            elif isinstance(metric_weight, list):
                metric_weight = np.array([metric_weight]).flatten()

            else:
                metric_weight = metric_weight.flatten()

            def weighting_func(state, circuit):
                return np.dot(
                    metric_weight,
                    np.array(
                        [met.evaluate(state, circuit) for met in self._metrics]
                    ).flatten(),
                )

            self.weighting_func = weighting_func
        elif callable(metric_weight):
            self.weighting_func = metric_weight

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
        val = self.weighting_func(state, circuit)
        self.increment()

        if self._inc % self.log_steps == 0:
            self.log.append(val)

        return val

    @property
    def per_metric_log(self):
        """
        The joint log of all metric functions

        :return: the log itself
        :rtype: dict (keys are metric class names, values are the logs)
        """
        m = {}
        for i, metric in enumerate(self._metrics):
            m[metric.__class__.__name__] = metric.log

        return m
