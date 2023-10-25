import pytest
from src.circuit.circuit_dag import CircuitDAG


@pytest.fixture(scope="function")
def dag():
    return CircuitDAG()
