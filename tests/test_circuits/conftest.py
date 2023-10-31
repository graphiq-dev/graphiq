import pytest
from graphiq.circuit.circuit_dag import CircuitDAG


@pytest.fixture(scope="function")
def dag():
    return CircuitDAG()
