import pytest
from src.metrics import *


def test_metric_base():
    test_metric = MetricBase()

    with pytest.raises(NotImplementedError):
        test_metric.evaluate()


# test infidelity
def test_infidelity():
    pass


# test trace distance
def test_trace_distance():
    pass


# test circuit depth
def test_circuit_depth():
    pass


# test metrics
def test_metrics():
    pass



