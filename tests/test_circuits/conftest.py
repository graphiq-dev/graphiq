import pytest
from src.circuit import CircuitDAG


@pytest.fixture(scope='function')
def dag():
    return CircuitDAG()
