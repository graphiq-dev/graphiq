import pytest
from src.circuit import RegisterCircuitDAG


@pytest.fixture(scope='function')
def dag():
    return RegisterCircuitDAG()
