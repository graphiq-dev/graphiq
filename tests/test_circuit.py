import pytest
import random

from src.circuit import CircuitDAG
from src.ops import Operation


@pytest.mark.parametrize("n_quantum, n_classical", [(1, 0), (0, 4), (3, 6), (24, 63)])
def test_initialization(n_quantum, n_classical):
    dag = CircuitDAG(n_quantum, n_classical)
    dag.validate()
    assert dag.n_quantum == n_quantum
    assert dag.n_classical == n_classical


def test_add_op_1():
    """
    Test single qubit gates
    """
    dag = CircuitDAG(2, 0)
    # retrieve internal information--this should not be done other than for testing
    op_q0_in = dag.dag.nodes['q0_in']['op']
    op_q1_in = dag.dag.nodes['q1_in']['op']
    op_q0_out = dag.dag.nodes['q0_out']['op']
    op_q1_out = dag.dag.nodes['q1_out']['op']

    op1 = Operation(q_registers=(0,))
    op2 = Operation(q_registers=(0,))
    op3 = Operation(q_registers=(0,))
    op4 = Operation(q_registers=(1,))
    dag.add(op1)
    dag.add(op2)
    dag.add(op3)
    dag.add(op4)

    # make sure all sources are Input operations, all sinks are Output operations
    dag.validate()


    op_order = dag.sequence()
    # check that topological order is correct
    assert op_order.index(op_q0_in) < op_order.index(op1) \
           < op_order.index(op2) < op_order.index(op3) < op_order.index(op_q0_out)
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
    dag = CircuitDAG(2, 2)

    op_q0_in = dag.dag.nodes['q0_in']['op']
    op_q1_in = dag.dag.nodes['q1_in']['op']
    op_c0_in = dag.dag.nodes['c0_in']['op']
    op_c1_in = dag.dag.nodes['c1_in']['op']
    op_q0_out = dag.dag.nodes['q0_out']['op']
    op_q1_out = dag.dag.nodes['q1_out']['op']
    op_c0_out = dag.dag.nodes['c0_out']['op']
    op_c1_out = dag.dag.nodes['c1_out']['op']

    op1 = Operation(q_registers=(1,))
    op2 = Operation(q_registers=(1, 0))
    op3 = Operation(q_registers=(1,), c_registers=(0,))
    op4 = Operation(c_registers=(1,))
    op5 = Operation(c_registers=(0, 1))
    op6 = Operation(q_registers=(0, 1), c_registers=(0, 1))
    op7 = Operation(q_registers=(0,))
    dag.add(op1)
    dag.add(op2)
    dag.add(op3)
    dag.add(op4)
    dag.add(op5)
    dag.add(op6)
    dag.add(op7)

    dag.validate()

    op_order = dag.sequence()
    assert op_order.index(op_q0_in) < op_order.index(op2)
    assert op_order.index(op_q1_in) < op_order.index(op1) < op_order.index(op2) \
           < op_order.index(op3) < op_order.index(op5) < op_order.index(op6) \
           < op_order.index(op7) < op_order.index(op_q0_out)
    assert op_order.index(op6) < op_order.index(op_q1_out)
    assert op_order.index(op6) < op_order.index(op_c0_out)
    assert op_order.index(op6) < op_order.index(op_c1_out)
    assert op_order.index(op_c0_in) < op_order.index(op3)
    assert op_order.index(op_c1_in) < op_order.index(op4) < op_order.index(op5)

    assert dag.n_quantum == 2
    assert dag.n_classical == 2
    assert dag.dag.number_of_nodes() == 15
    assert dag.dag.number_of_edges() == 16


def test_validate_correct():
    """
    Intentionally breaks DAG circuit structure to see that we can detect errors
    """
    dag = CircuitDAG(2, 2)

    op1 = Operation(q_registers=(1,))
    op2 = Operation(q_registers=(1, 0))
    op3 = Operation(q_registers=(1,), c_registers=(0,))
    dag.add(op1)
    dag.add(op2)
    dag.add(op3)

    # sabotage graph -- note that we should not directly manipulate the DiGraph except in tests
    dag.dag.remove_edge(3, 'c0_out')
    with pytest.raises(AssertionError):
        dag.validate()


@pytest.mark.parametrize("seed", [0, 4, 20, 9040])
def test_random_graph(seed):
    random.seed(seed)  # ensures tests are reproducible
    q = random.randint(1, 7)
    c = random.randint(0, 7)
    dag = CircuitDAG(q, c)
    dag.validate()
    for i in range(200):  # we'll apply 200 random gates
        q_register_num_max = random.randint(1, q)
        c_register_num_max = random.randint(0, c)
        q_registers = tuple(set([random.randint(0, q - 1) for _ in range(q_register_num_max)]))
        c_registers = tuple(set([random.randint(0, c - 1) for _ in range(c_register_num_max)]))

        dag.add(Operation(q_registers=q_registers, c_registers=c_registers))

    dag.validate()


def test_dynamic_registers_1():
    """
    Check with single-register gates only
    """
    dag = CircuitDAG(1, 0)
    op1 = Operation(q_registers=(1,))
    op2 = Operation(q_registers=(2,))
    op3 = Operation(c_registers=(5,))
    dag.add(op1)
    dag.validate()
    dag.add(op2)
    dag.validate()
    dag.add(op3)
    dag.validate()

    # retrieve internal information--this should not be done other than for testing
    op_q0_in = dag.dag.nodes['q0_in']['op']
    op_q1_in = dag.dag.nodes['q1_in']['op']
    op_q2_in = dag.dag.nodes['q2_in']['op']
    op_c5_in = dag.dag.nodes['c5_in']['op']

    op_q0_out = dag.dag.nodes['q0_out']['op']
    op_q1_out = dag.dag.nodes['q1_out']['op']
    op_q2_out = dag.dag.nodes['q2_out']['op']
    op_c5_out = dag.dag.nodes['c5_out']['op']

    # check topological order
    op_order = dag.sequence()
    assert op_order.index(op_q0_in) < op_order.index(op_q0_out)
    assert op_order.index(op_q1_in) < op_order.index(op1) < op_order.index(op_q1_out)
    assert op_order.index(op_q2_in) < op_order.index(op2) < op_order.index(op_q2_out)
    assert op_order.index(op_c5_in) < op_order.index(op3) < op_order.index(op_c5_out)

    assert dag.n_quantum == 3
    assert dag.n_classical == 1
    assert dag.dag.number_of_nodes() == 11
    assert dag.dag.number_of_edges() == 7


def test_dynamic_register_2():
    """
    Same test, allowing 2 qubit gates
    """
    dag = CircuitDAG(1, 0)
    op1 = Operation(q_registers=(1, 2))
    op2 = Operation(q_registers=(2,))
    op3 = Operation(q_registers=(0,), c_registers=(5, 0))
    dag.add(op1)
    dag.add(op2)
    dag.add(op3)
    dag.validate()

    # retrieve internal information--this should not be done other than for testing
    op_q0_in = dag.dag.nodes['q0_in']['op']
    op_q1_in = dag.dag.nodes['q1_in']['op']
    op_q2_in = dag.dag.nodes['q2_in']['op']
    op_c0_in = dag.dag.nodes['c0_in']['op']
    op_c5_in = dag.dag.nodes['c5_in']['op']

    op_q0_out = dag.dag.nodes['q0_out']['op']
    op_q1_out = dag.dag.nodes['q1_out']['op']
    op_q2_out = dag.dag.nodes['q2_out']['op']
    op_c0_out = dag.dag.nodes['c0_out']['op']
    op_c5_out = dag.dag.nodes['c5_out']['op']

    # check topological order
    op_order = dag.sequence()
    assert op_order.index(op_c0_in) < op_order.index(op3) < op_order.index(op_c0_out)
    assert op_order.index(op_c5_in) < op_order.index(op3) < op_order.index(op_c5_out)
    assert op_order.index(op_q0_in) < op_order.index(op3) < op_order.index(op_q0_out)
    assert op_order.index(op_q1_in) < op_order.index(op1) < op_order.index(op_q1_out)
    assert op_order.index(op_q2_in) < op_order.index(op1) < op_order.index(op2) < op_order.index(op_q2_out)

    assert dag.n_quantum == 3
    assert dag.n_classical == 2
    assert dag.dag.number_of_nodes() == 13
    assert dag.dag.number_of_edges() == 11


@pytest.mark.parametrize("seed", [0, 4, 20, 9040])
def test_random_graph(seed):
    random.seed(seed)
    dag = CircuitDAG(1, 1)
    dag.validate()
    for i in range(200):  # we'll apply 200 random gates
        q_register_num = random.randint(1, 5)
        c_register_num = random.randint(1, 5)

        q_registers = tuple(set([random.randint(0, 150) for _ in range(q_register_num)]))
        c_registers = tuple(set([random.randint(0, 120) for _ in range(c_register_num)]))
        dag.add(Operation(q_registers=q_registers, c_registers=c_registers))

    dag.validate()



