"""
Classes for compute metrics of
"""

from abc import ABC, abstractmethod
import numpy as np
import time


class MetricBase(ABC):
    """
    Base class for a metric
    """

    log_steps = 1

    def __init__(self, *args, **kwargs):
        self.name = "base"
        self.differentiable = False

        self.log = []  # will store the metric evaluations
        self._inc = 0  #

    @abstractmethod
    def evaluate(self, state, circuit):
        raise NotImplementedError("Please use an inherited class, not the base metric class")

    def increment(self):
        self._inc += 1


class MetricFidelity(MetricBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.differentiable = False

    def evaluate(self, state, circuit):
        val = 1.0
        self.increment()

        if self._inc % self.log_steps == 0:
            self.log.append(val)

        return val


class MetricCircuitDepth(MetricBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.differentiable = False

    def evaluate(self, state, circuit):
        val = 1.0
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
