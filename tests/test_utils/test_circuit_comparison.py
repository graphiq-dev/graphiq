import pytest
from src.circuit import CircuitDAG
from benchmarks.circuits import *
from src.ops import *
from src.utils.circuit_comparison import circuit_is_isomorphic


def test_circuit_is_isomorphic_ghz3():
    circuit1, state1 = ghz3_state_circuit()
    circuit2, state2 = ghz3_state_circuit()

    assert circuit_is_isomorphic(circuit1, circuit2)


@pytest.mark.parametrize(
    "circuit1_params, circuit2_params, result", [
        ((1, 1), (1, 1), True),
        ((1, 2), (2, 1), False),
        ((3, 2), (3, 2), True),
    ]
)
def test_circuit_is_isomorphic_empty(circuit1_params, circuit2_params, result):
    circuit1 = CircuitDAG(n_photon=circuit1_params[0], n_emitter=circuit1_params[1])
    circuit2 = CircuitDAG(n_photon=circuit2_params[0], n_emitter=circuit2_params[1])

    assert circuit_is_isomorphic(circuit1, circuit2) == result


def test_circuit_is_isomorphic_relabeling_1():
    circuit1 = CircuitDAG(n_photon=2)
    circuit2 = CircuitDAG(n_photon=2)

    circuit1.add(Hadamard(reg_type='p', register=0))
    circuit2.add(Hadamard(reg_type='p', register=1))

    assert not circuit_is_isomorphic(circuit1, circuit2)


def test_circuit_is_isomorphic_relabeling_2():
    circuit1 = CircuitDAG(n_emitter=2)
    circuit2 = CircuitDAG(n_emitter=2)

    circuit1.add(Hadamard(reg_type='e', register=0))
    circuit2.add(Hadamard(reg_type='e', register=1))

    assert circuit_is_isomorphic(circuit1, circuit2)


def test_circuit_is_isomorphic_relabeling_3():
    circuit1 = CircuitDAG(n_photon=2)
    circuit2 = CircuitDAG(n_photon=2)

    circuit1.add(CNOT(target=0, target_type='p', control=1, control_type='p'))
    circuit2.add(CNOT(target=1, target_type='p', control=0, control_type='p'))

    assert not circuit_is_isomorphic(circuit1, circuit2)


def test_circuit_is_isomorphic_relabeling_4():
    circuit1 = CircuitDAG(n_photon=1, n_emitter=2)
    circuit2 = CircuitDAG(n_photon=1, n_emitter=2)

    circuit1.add(CNOT(target=0, target_type='p', control=0, control_type='e'))
    circuit2.add(CNOT(target=0, target_type='p', control=1, control_type='e'))

    assert circuit_is_isomorphic(circuit1, circuit2)


def test_circuit_is_isomorphic_relabeling_5():
    circuit1 = CircuitDAG(n_photon=2, n_emitter=2)
    circuit2 = CircuitDAG(n_photon=2, n_emitter=2)

    circuit1.add(CNOT(target=0, target_type='p', control=0, control_type='e'))
    circuit2.add(CNOT(target=1, target_type='p', control=1, control_type='e'))

    assert not circuit_is_isomorphic(circuit1, circuit2)


def test_circuit_is_isomorphic_relabeling_6():
    circuit1 = CircuitDAG(n_photon=2, n_emitter=2)
    circuit2 = CircuitDAG(n_photon=2, n_emitter=2)

    circuit1.add(CNOT(target=0, target_type='p', control=0, control_type='e'))
    circuit1.add(CNOT(target=1, target_type='p', control=0, control_type='e'))
    circuit1.add(CNOT(target=0, target_type='e', control=1, control_type='e'))

    circuit2.add(CNOT(target=0, target_type='p', control=1, control_type='e'))
    circuit2.add(CNOT(target=1, target_type='p', control=1, control_type='e'))
    circuit2.add(CNOT(target=0, target_type='e', control=1, control_type='e'))

    assert not circuit_is_isomorphic(circuit1, circuit2)


def test_circuit_is_isomorphic_relabeling_7():
    circuit1 = CircuitDAG(n_photon=2, n_emitter=2)
    circuit2 = CircuitDAG(n_photon=2, n_emitter=2)

    circuit1.add(CNOT(target=0, target_type='p', control=0, control_type='e'))
    circuit1.add(CNOT(target=1, target_type='p', control=0, control_type='e'))
    circuit1.add(CNOT(target=1, target_type='e', control=0, control_type='e'))

    circuit2.add(CNOT(target=0, target_type='p', control=1, control_type='e'))
    circuit2.add(CNOT(target=1, target_type='p', control=1, control_type='e'))
    circuit2.add(CNOT(target=0, target_type='e', control=1, control_type='e'))

    assert circuit_is_isomorphic(circuit1, circuit2)


def test_circuit_is_isomorphic_relabeling_8():
    circuit1 = CircuitDAG(n_photon=2, n_emitter=2)
    circuit2 = CircuitDAG(n_photon=2, n_emitter=2)

    circuit1.add(CNOT(target=0, target_type='p', control=0, control_type='e'))
    circuit1.add(CNOT(target=1, target_type='p', control=0, control_type='e'))
    circuit1.add(CNOT(target=1, target_type='e', control=0, control_type='e'))

    circuit2.add(CNOT(target=0, target_type='p', control=0, control_type='e'))
    circuit2.add(CNOT(target=1, target_type='p', control=0, control_type='e'))
    circuit2.add(CNOT(target=0, target_type='e', control=1, control_type='e'))

    assert not circuit_is_isomorphic(circuit1, circuit2)
