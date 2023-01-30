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
    "circuit1_params, circuit2_params, result",
    [
        ((1, 1), (1, 1), True),
        ((1, 2), (2, 1), False),
        ((3, 2), (3, 2), True),
    ],
)
def test_circuit_is_isomorphic_empty(circuit1_params, circuit2_params, result):
    circuit1 = CircuitDAG(n_photon=circuit1_params[0], n_emitter=circuit1_params[1])
    circuit2 = CircuitDAG(n_photon=circuit2_params[0], n_emitter=circuit2_params[1])

    assert circuit_is_isomorphic(circuit1, circuit2) == result


def test_circuit_is_isomorphic_1():
    circuit1 = CircuitDAG(n_photon=2)
    circuit2 = CircuitDAG(n_photon=2)

    circuit1.add(Hadamard(reg_type="p", register=0))
    circuit2.add(Hadamard(reg_type="p", register=1))

    assert circuit_is_isomorphic(circuit1, circuit2)


def test_circuit_is_isomorphic_2():
    circuit1 = CircuitDAG(n_emitter=2)
    circuit2 = CircuitDAG(n_emitter=2)

    circuit1.add(Hadamard(reg_type="e", register=0))
    circuit2.add(Hadamard(reg_type="e", register=1))

    assert circuit_is_isomorphic(circuit1, circuit2)


def test_circuit_is_isomorphic_3():
    # swap photons
    circuit_1 = CircuitDAG(n_emitter=1, n_photon=3)
    circuit_1.add(CNOT(0, "e", 0, "p"))
    circuit_1.add(CNOT(0, "e", 1, "p"))
    circuit_1.add(CNOT(0, "e", 2, "p"))

    circuit_2 = CircuitDAG(n_emitter=1, n_photon=3)
    circuit_2.add(CNOT(0, "e", 0, "p"))
    circuit_2.add(CNOT(0, "e", 2, "p"))
    circuit_2.add(CNOT(0, "e", 1, "p"))

    assert circuit_is_isomorphic(circuit_1, circuit_2)


def test_circuit_is_isomorphic_4():
    # %% 2 emitters and 2 photons, swap only emitters
    circuit_1 = CircuitDAG(n_emitter=2, n_photon=1, n_classical=0)
    circuit_1.add(Hadamard(register=0, reg_type="e"))
    circuit_1.add(Hadamard(register=1, reg_type="e"))
    circuit_1.add(CNOT(control=0, control_type="e", target=0, target_type="p"))
    circuit_1.add(CNOT(control=0, control_type="e", target=1, target_type="e"))
    circuit_1.add(CNOT(control=1, control_type="e", target=1, target_type="p"))
    circuit_1.validate()

    circuit_2 = CircuitDAG(n_emitter=2, n_photon=1, n_classical=0)
    circuit_2.add(Hadamard(register=1, reg_type="e"))
    circuit_2.add(Hadamard(register=0, reg_type="e"))
    circuit_2.add(CNOT(control=1, control_type="e", target=0, target_type="p"))
    circuit_2.add(CNOT(control=1, control_type="e", target=0, target_type="e"))
    circuit_2.add(CNOT(control=0, control_type="e", target=1, target_type="p"))
    circuit_2.validate()

    assert circuit_is_isomorphic(circuit_1, circuit_2)


def test_circuit_is_isomorphic_5():
    # %% 2 emitters and 2 photons, swap all emitters and photons
    circuit_1 = CircuitDAG(n_emitter=2, n_photon=1, n_classical=0)
    circuit_1.add(Hadamard(register=0, reg_type="e"))
    circuit_1.add(Hadamard(register=1, reg_type="e"))
    circuit_1.add(CNOT(control=0, control_type="e", target=0, target_type="p"))
    circuit_1.add(CNOT(control=0, control_type="e", target=1, target_type="e"))
    circuit_1.add(CNOT(control=1, control_type="e", target=1, target_type="p"))

    circuit_2 = CircuitDAG(n_emitter=2, n_photon=1, n_classical=0)
    circuit_2.add(Hadamard(register=1, reg_type="e"))
    circuit_2.add(Hadamard(register=0, reg_type="e"))
    circuit_2.add(CNOT(control=1, control_type="e", target=1, target_type="p"))
    circuit_2.add(CNOT(control=1, control_type="e", target=0, target_type="e"))
    circuit_2.add(CNOT(control=0, control_type="e", target=0, target_type="p"))

    assert circuit_is_isomorphic(circuit_1, circuit_2)


def test_circuit_is_isomorphic_6():
    # swap e0 -> e1, e1 -> e2
    circuit_1 = CircuitDAG(n_emitter=3, n_photon=3, n_classical=0)
    circuit_1.add(Hadamard(register=0, reg_type="e"))
    circuit_1.add(Hadamard(register=1, reg_type="e"))
    circuit_1.add(Hadamard(register=2, reg_type="e"))
    circuit_1.add(CNOT(control=0, control_type="e", target=0, target_type="p"))
    circuit_1.add(CNOT(control=0, control_type="e", target=1, target_type="e"))
    circuit_1.add(CNOT(control=1, control_type="e", target=1, target_type="p"))
    circuit_1.add(CNOT(control=1, control_type="e", target=2, target_type="e"))
    circuit_1.add(CNOT(control=2, control_type="e", target=2, target_type="p"))

    circuit_2 = CircuitDAG(n_emitter=3, n_photon=3, n_classical=0)
    circuit_2.add(Hadamard(register=0, reg_type="e"))
    circuit_2.add(Hadamard(register=1, reg_type="e"))
    circuit_2.add(Hadamard(register=2, reg_type="e"))
    circuit_2.add(CNOT(control=1, control_type="e", target=0, target_type="p"))
    circuit_2.add(CNOT(control=1, control_type="e", target=2, target_type="e"))
    circuit_2.add(CNOT(control=2, control_type="e", target=1, target_type="p"))
    circuit_2.add(CNOT(control=2, control_type="e", target=0, target_type="e"))
    circuit_2.add(CNOT(control=0, control_type="e", target=2, target_type="p"))

    assert circuit_is_isomorphic(circuit_1, circuit_2)
