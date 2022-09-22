import pytest
import random

from src.circuit import CircuitDAG
import src.ops as ops
from tests.test_flags import visualization


@pytest.mark.parametrize(
    "n_emitter, n_photon, n_classical", [(1, 0, 0), (0, 4, 2), (3, 6, 1), (24, 2, 63)]
)
def test_initialization(n_emitter, n_photon, n_classical):
    dag = CircuitDAG(n_emitter=n_emitter, n_photon=n_photon, n_classical=n_classical)
    dag.validate()
    assert dag.n_quantum == n_photon + n_emitter
    assert dag.n_photons == n_photon
    assert dag.n_emitters == n_emitter
    assert dag.n_classical == n_classical


def test_add_1():
    dag = CircuitDAG(n_emitter=1, n_classical=0)
    dag.add(ops.Hadamard(register=0))
    dag.validate()


def test_add_2():
    dag = CircuitDAG(n_emitter=2, n_classical=0)
    dag.add(ops.CNOT(control=0, target=1))
    dag.add(ops.CNOT(control=0, target=1))
    dag.validate()


def test_add_3():
    dag = CircuitDAG(n_emitter=2, n_photon=1, n_classical=0)
    dag.add(ops.CNOT(control=0, control_type="e", target=0, target_type="p"))
    dag.add(ops.CNOT(control=0, control_type="e", target=1, target_type="e"))
    dag.validate()


def test_add_op_1():
    """
    Test single qubit gates
    """
    dag = CircuitDAG(n_emitter=2, n_classical=0)
    # retrieve internal information--this should not be done other than for testing
    op_q0_in = dag.dag.nodes["e0_in"]["op"]
    op_q1_in = dag.dag.nodes["e1_in"]["op"]
    op_q0_out = dag.dag.nodes["e0_out"]["op"]
    op_q1_out = dag.dag.nodes["e1_out"]["op"]

    op1 = ops.Hadamard(register=0, reg_type="e")
    op2 = ops.Hadamard(register=1, reg_type="e")
    op3 = ops.Hadamard(register=0, reg_type="e")
    op4 = ops.Hadamard(register=1, reg_type="e")
    dag.add(op1)
    dag.add(op2)
    dag.add(op3)
    dag.add(op4)

    dag.validate()

    op_order = dag.sequence()
    # check that topological order is correct
    assert (
        op_order.index(op_q0_in)
        < op_order.index(op1)
        < op_order.index(op2)
        < op_order.index(op3)
        < op_order.index(op_q0_out)
    )
    assert op_order.index(op_q1_in) < op_order.index(op4) < op_order.index(op_q1_out)

    # check that the numbers of nodes / edges are as expected
    assert dag.n_quantum == 2
    assert dag.n_classical == 0
    assert dag.dag.number_of_nodes() == 8
    assert dag.dag.number_of_edges() == 6


def test_add_op2():
    """
    Test multi register gates
    """
    dag = CircuitDAG(n_emitter=1, n_photon=1, n_classical=2)

    op_e0_in = dag.dag.nodes["e0_in"]["op"]
    op_p0_in = dag.dag.nodes["p0_in"]["op"]
    op_c0_in = dag.dag.nodes["c0_in"]["op"]
    op_c1_in = dag.dag.nodes["c1_in"]["op"]
    op_e0_out = dag.dag.nodes["e0_out"]["op"]
    op_p0_out = dag.dag.nodes["p0_out"]["op"]
    op_c0_out = dag.dag.nodes["c0_out"]["op"]
    op_c1_out = dag.dag.nodes["c1_out"]["op"]

    op1 = ops.CNOT(control=0, control_type="e", target=1, target_type="e")
    op2 = ops.CNOT(control=0, control_type="e", target=1, target_type="e")
    op3 = ops.CNOT(control=0, control_type="p", target=1, target_type="p")
    op4 = ops.CNOT(control=0, control_type="e", target=1, target_type="e")
    op5 = ops.CNOT(control=0, control_type="e", target=1, target_type="e")
    op6 = ops.CNOT(control=0, control_type="e", target=1, target_type="e")
    op7 = ops.CNOT(control=0, control_type="p", target=1, target_type="p")
    dag.add(op1)
    dag.add(op2)
    dag.add(op3)
    dag.add(op4)
    dag.add(op5)
    dag.add(op6)
    dag.add(op7)

    dag.validate()

    op_order = dag.sequence()
    assert op_order.index(op_e0_in) < op_order.index(op1) < op_order.index(op2)
    assert op_order.index(op_p0_in) < op_order.index(op2)
    assert (
        op_order.index(op3)
        < op_order.index(op2)
        < op_order.index(op7)
        < op_order.index(op4)
        < op_order.index(op5)
        < op_order.index(op6)
        < op_order.index(op_e0_out)
    )
    assert op_order.index(op_c0_in) < op_order.index(op3)
    assert op_order.index(op_c1_in) < op_order.index(op6)
    assert op_order.index(op_p0_out) < op_order.index(op5)
    assert op_order.index(op_c0_out) < op_order.index(op6)
    assert op_order.index(op_c1_out) < op_order.index(op6)

    assert dag.n_quantum == 4
    assert dag.n_photons == 2
    assert dag.n_emitters == 2
    assert dag.n_classical == 2
    assert dag.dag.number_of_nodes() == 19
    assert dag.dag.number_of_edges() == 20


def test_validate_correct():
    """
    Intentionally breaks DAG circuit structure to see that we can detect errors
    """
    dag = CircuitDAG(n_emitter=2, n_classical=2)

    op1 = ops.CNOT(control=0, control_type="e", target=1, target_type="e")
    op2 = ops.CNOT(control=0, control_type="e", target=1, target_type="e")
    op3 = ops.CNOT(control=0, control_type="p", target=1, target_type="p")
    dag.add(op1)
    dag.add(op2)
    dag.add(op3)

    # sabotage graph -- note that we should not directly manipulate the DiGraph except in tests
    dag.dag.remove_edge(3, "p0_out")
    with pytest.raises(RuntimeError):
        dag.validate()


@pytest.mark.parametrize("seed", [0, 4, 20, 9040])
def test_random_graph(seed):
    random.seed(seed)  # ensures tests are reproducible
    q = random.randint(1, 7)
    c = random.randint(0, 7)
    dag = CircuitDAG(n_emitter=q, n_classical=c)
    dag.validate()
    for i in range(200):  # we'll apply 200 random gates
        q_register_num_max = random.randint(1, q)
        c_register_num_max = random.randint(0, c)
        q_registers = tuple(
            set([random.randint(0, q - 1) for _ in range(q_register_num_max)])
        )
        c_registers = tuple(
            set([random.randint(0, c - 1) for _ in range(c_register_num_max)])
        )

        dag.add(ops.Hadamard(register=0, reg_type="e"))

    dag.validate()


def test_dynamic_registers_1():
    """
    Check with single-register gates only
    """
    dag = CircuitDAG(n_emitter=1, n_classical=0)
    op1 = ops.Hadamard(register=1, reg_type="e")
    op2 = ops.Hadamard(register=2, reg_type="e")
    op3 = ops.Hadamard(register=0, reg_type="p")
    dag.add(op1)
    dag.validate()
    dag.add(op2)
    dag.validate()
    dag.add(op3)
    dag.validate()
    dag.draw_dag()
    # retrieve internal information--this should not be done other than for testing
    op_q0_in = dag.dag.nodes["e0_in"]["op"]
    op_q1_in = dag.dag.nodes["e1_in"]["op"]
    op_q2_in = dag.dag.nodes["e2_in"]["op"]
    op_c5_in = dag.dag.nodes["p0_in"]["op"]

    op_q0_out = dag.dag.nodes["e0_out"]["op"]
    op_q1_out = dag.dag.nodes["e1_out"]["op"]
    op_q2_out = dag.dag.nodes["e2_out"]["op"]
    op_c5_out = dag.dag.nodes["p0_out"]["op"]

    # check topological order
    op_order = dag.sequence()
    assert op_order.index(op_q0_in) < op_order.index(op_q0_out)
    assert op_order.index(op_q1_in) < op_order.index(op1) < op_order.index(op_q1_out)
    assert op_order.index(op_q2_in) < op_order.index(op2) < op_order.index(op_q2_out)
    assert op_order.index(op_c5_in) < op_order.index(op3) < op_order.index(op_c5_out)

    assert dag.n_quantum == 4
    assert dag.n_classical == 0
    assert dag.dag.number_of_nodes() == 11
    assert dag.dag.number_of_edges() == 7


def test_continuous_indices_registers():
    """
    Check with single-register gates only
    """
    dag = CircuitDAG(n_emitter=1, n_classical=0)
    op1 = ops.Hadamard(register=1, reg_type="e")
    op2 = ops.Hadamard(register=0, reg_type="e")
    op3 = ops.Hadamard(register=5, reg_type="p")
    dag.add(op1)
    dag.validate()
    dag.add(op2)
    dag.validate()
    with pytest.raises(ValueError):
        dag.add(op3)


def test_dynamic_register_2():
    """
    Same test, allowing 2 qubit gates
    """
    dag = CircuitDAG(n_emitter=1, n_classical=0)
    op1 = ops.CNOT(control=0, control_type="e", target=1, target_type="e")
    op2 = ops.CNOT(control=1, control_type="e", target=2, target_type="e")
    op3 = ops.CNOT(control=0, control_type="p", target=1, target_type="p")
    dag.add(op1)
    dag.add(op2)
    dag.add(op3)
    dag.validate()

    # retrieve internal information--this should not be done other than for testing
    op_q0_in = dag.dag.nodes["e0_in"]["op"]
    op_q1_in = dag.dag.nodes["e1_in"]["op"]
    op_q2_in = dag.dag.nodes["e2_in"]["op"]
    op_c0_in = dag.dag.nodes["p0_in"]["op"]
    op_c5_in = dag.dag.nodes["p0_in"]["op"]

    op_q0_out = dag.dag.nodes["e0_out"]["op"]
    op_q1_out = dag.dag.nodes["e1_out"]["op"]
    op_q2_out = dag.dag.nodes["e2_out"]["op"]
    op_c0_out = dag.dag.nodes["p0_out"]["op"]
    op_c5_out = dag.dag.nodes["p0_out"]["op"]

    # check topological order
    op_order = dag.sequence()
    assert op_order.index(op_c0_in) < op_order.index(op3) < op_order.index(op_c0_out)
    assert op_order.index(op_c5_in) < op_order.index(op3) < op_order.index(op_c5_out)
    assert op_order.index(op_q0_in) < op_order.index(op3) < op_order.index(op_q0_out)
    assert op_order.index(op_q1_in) < op_order.index(op1) < op_order.index(op_q1_out)
    assert (
        op_order.index(op_q2_in)
        < op_order.index(op1)
        < op_order.index(op2)
        < op_order.index(op_q2_out)
    )

    assert dag.n_quantum == 5
    assert dag.n_classical == 0
    assert dag.dag.number_of_nodes() == 13
    assert dag.dag.number_of_edges() == 11


@pytest.mark.parametrize("seed", [0, 4, 20, 9040])
def test_random_graph(seed):
    random.seed(seed)
    dag = CircuitDAG(n_emitter=150, n_classical=120)
    dag.validate()
    for i in range(200):  # we'll apply 200 random gates
        q_register_num = random.randint(1, 5)
        c_register_num = random.randint(1, 5)

        q_registers = tuple(
            set([random.randint(0, 150) for _ in range(q_register_num)])
        )
        c_registers = tuple(
            set([random.randint(0, 120) for _ in range(c_register_num)])
        )
        dag.add(ops.Hadamard(register=0, reg_type="e"))

    dag.validate()


# Test register implementations


def test_add_register_1():
    """
    Test the fact we can't add registers of size 0
    """
    dag = CircuitDAG()
    dag.validate()
    with pytest.raises(ValueError):
        dag.add_emitter_register(2)
    dag.add_emitter_register(1)
    dag.validate()


def test_expand_register_1():
    """
    Test the fact we can't add expand register size for these graphs
    """
    dag = CircuitDAG(n_emitter=2, n_classical=2)
    dag.validate()
    with pytest.raises(ValueError):
        dag.expand_emitter_register(0, 2)
    with pytest.raises(ValueError):
        dag.expand_classical_register(0, 2)
    dag.validate()


def test_add_register_2():
    dag = CircuitDAG(n_emitter=2, n_classical=2)
    dag.validate()
    dag.add_emitter_register()
    dag.add(ops.Hadamard(register=0, reg_type="e"))
    dag.validate()
    dag.add(ops.Hadamard(register=1, reg_type="e"))
    with pytest.raises(ValueError):
        dag.add_classical_register(2)
    dag.add_classical_register(1)
    dag.validate()

    assert dag.n_quantum == 3
    assert dag.n_classical == 3
    assert dag.dag.number_of_nodes() == 14
    assert dag.dag.number_of_edges() == 8


def test_register_setting():
    """
    Note: this should never be done externally (only internally to the Circuit classes)
    Nevertheless we test it here just in case
    """
    dag = CircuitDAG(n_emitter=2, n_classical=2)
    dag.emitter_registers = [1, 1]
    dag.c_registers = [1, 1]
    with pytest.raises(ValueError):
        dag.emitter_registers = [0, 1]
    with pytest.raises(ValueError):
        dag.c_registers = [2, 1]


def test_sequence_unwinding():
    """Test that the sequence unwrapping with the Wrapper operation works"""
    gates = [ops.Hadamard, ops.Phase, ops.Hadamard, ops.Phase, ops.Identity]
    operation = ops.OneQubitGateWrapper(gates, register=0, reg_type="e")
    dag = CircuitDAG(n_emitter=1, n_photon=1, n_classical=0)
    dag.add(operation)
    dag.add(ops.CNOT(control=0, control_type="e", target=0, target_type="p"))

    # sequence without unwrapping
    op_e0_in = dag.dag.nodes["e0_in"]["op"]
    op_p0_in = dag.dag.nodes["p0_in"]["op"]
    op_e0_out = dag.dag.nodes["e0_out"]["op"]
    op_p0_out = dag.dag.nodes["p0_out"]["op"]

    op2 = dag.dag.nodes[2]["op"]

    op_order = dag.sequence()
    assert (
        op_order.index(op_e0_in)
        < op_order.index(operation)
        < op_order.index(op2)
        < op_order.index(op_e0_out)
    )
    assert op_order.index(op_p0_in) < op_order.index(op2) < op_order.index(op_p0_out)

    # sequence with unwrapping
    op_order = dag.sequence(unwrapped=True)
    op_class_order = [
        op.__class__
        for op in op_order
        if not isinstance(op, ops.InputOutputOperationBase)
    ]
    assert op_class_order == [
        ops.Identity,
        ops.Phase,
        ops.Hadamard,
        ops.Phase,
        ops.Hadamard,
        ops.CNOT,
    ]


@visualization
def test_visualization_1(dag):
    dag.validate()
    dag.draw_dag()


@visualization
def test_visualization_2():
    circuit2 = CircuitDAG(n_emitter=3, n_classical=2)
    circuit2.validate()
    circuit2.draw_dag()


@visualization
def test_visualization_3():
    circuit3 = CircuitDAG(n_emitter=2, n_classical=0)
    circuit3.add(ops.CNOT(control=0, control_type="e", target=1, target_type="e"))
    circuit3.validate()
    circuit3.add(ops.Hadamard(register=1, reg_type="e"))
    circuit3.validate()
    circuit3.draw_dag()


@visualization
def test_visualization_4():
    circuit4 = CircuitDAG(n_emitter=3, n_classical=3)
    circuit4.add(ops.CNOT(control=0, control_type="e", target=1, target_type="e"))
    circuit4.add(ops.CNOT(control=0, control_type="e", target=2, target_type="e"))
    circuit4.add(ops.Hadamard(register=2, reg_type="e"))
    circuit4.add(ops.Phase(register=0, reg_type="e"))
    circuit4.validate()
    circuit4.draw_dag()


@visualization
def test_visualization_5():
    # test visualization when dynamic dealing with register number (copied from test_circuit)
    dag = CircuitDAG(n_emitter=1, n_classical=0)
    op1 = ops.CNOT(control=0, control_type="e", target=1, target_type="e")
    op2 = ops.Hadamard(register=1, reg_type="e")
    op3 = ops.Phase(register=0, reg_type="e")
    dag.add(op1)
    dag.add(op2)
    dag.add(op3)
    dag.validate()
    dag.draw_dag()


@visualization
def test_visualization_unwrapped_1():
    """Test that visualization works with the Wrapper operation"""
    gates = [ops.Hadamard, ops.Phase, ops.Hadamard, ops.Phase, ops.Identity]
    operation = ops.OneQubitGateWrapper(gates, register=0, reg_type="e")
    dag = CircuitDAG(n_emitter=1, n_photon=1, n_classical=0)
    dag.add(operation)
    dag.add(ops.CNOT(control=0, control_type="e", target=0, target_type="p"))
    dag.draw_dag()
    try:
        dag.draw_circuit()
    except Exception as e:
        print(dag.to_openqasm())
        raise e


@visualization
def test_visualization_unwrapped_2():
    """Test that visualization works with the Wrapper operation"""
    gates = [ops.Hadamard, ops.Phase, ops.Hadamard, ops.Phase, ops.Identity]
    operation = ops.OneQubitGateWrapper(gates, register=0, reg_type="e")
    dag = CircuitDAG(n_emitter=1, n_photon=1, n_classical=0)
    dag.add(operation)
    dag.add(ops.CNOT(control=0, control_type="e", target=0, target_type="p"))
    dag.add(
        ops.OneQubitGateWrapper(
            [ops.SigmaX, ops.SigmaY, ops.SigmaZ, ops.Phase], register=0, reg_type="e"
        )
    )
    dag.draw_dag()
    try:
        dag.draw_circuit()
    except Exception as e:
        print(dag.to_openqasm())
        raise e
