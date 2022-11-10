import pytest
from src.metrics import *
from benchmarks.circuits import *
from src.state import QuantumState


def test_metric_base():
    test_metric = MetricBase()

    with pytest.raises(NotImplementedError):
        test_metric.evaluate()


# test infidelity
def test_infidelity():
    circuit, state = ghz3_state_circuit()
    metrics = Infidelity(target=state)

    linear_cluster_3_graph = nx.Graph([(1, 2), (2, 3)])
    q_state = QuantumState(
        3, [linear_cluster_3_graph], representation=["graph"]
    )

    with pytest.raises(ValueError, match="Cannot compute the infidelity."):
        metrics.evaluate(q_state, circuit)


# test trace distance
def test_trace_distance():
    pass


# test circuit depth
def test_circuit_depth():
    circuit, state = ghz3_state_circuit()
    metrics = CircuitDepth()

    result = metrics.evaluate(state, circuit)
    assert result == 7


# test metrics
def test_metrics():
    pass



