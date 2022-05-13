import pytest
import random

from src.circuit import CircuitDAG
from src.ops import Operation


@pytest.mark.parametrize("n_qudit, n_cbit", [(1, 0), (0, 4), (3, 6), (24, 63)])
def test_initialization(n_qudit, n_cbit):
    dag = CircuitDAG(n_qudit, n_cbit)
    dag.validate()


def test_add_op_1():
    """
    Test single qubit gates
    """
    dag = CircuitDAG(2, 0)
    # retrieve internal information--this should not be done other than for testing
    op_q0_in = dag.DAG.nodes['q0_in']['op']
    op_q1_in = dag.DAG.nodes['q1_in']['op']
    op_q0_out = dag.DAG.nodes['q0_out']['op']
    op_q1_out = dag.DAG.nodes['q1_out']['op']

    op1 = Operation(qudits=(0,))
    op2 = Operation(qudits=(0,))
    op3 = Operation(qudits=(0,))
    op4 = Operation(qudits=(1,))
    dag.add_op(op1)
    dag.add_op(op2)
    dag.add_op(op3)
    dag.add_op(op4)

    # make sure all sources are Input operations, all sinks are Output operations
    dag.validate()

    op_order = dag.operation_list()
    # check that topological order is correct
    assert op_order.index(op_q0_in) < op_order.index(op1) \
           < op_order.index(op2) < op_order.index(op3) < op_order.index(op_q0_out)
    assert op_order.index(op_q1_in) < op_order.index(op4) < op_order.index(op_q1_out)


def test_add_op2():
    """
    Test multi register gates
    """
    dag = CircuitDAG(2, 2)

    op_q0_in = dag.DAG.nodes['q0_in']['op']
    op_q1_in = dag.DAG.nodes['q1_in']['op']
    op_c0_in = dag.DAG.nodes['c0_in']['op']
    op_c1_in = dag.DAG.nodes['c1_in']['op']
    op_q0_out = dag.DAG.nodes['q0_out']['op']
    op_q1_out = dag.DAG.nodes['q1_out']['op']
    op_c0_out = dag.DAG.nodes['c0_out']['op']
    op_c1_out = dag.DAG.nodes['c1_out']['op']

    op1 = Operation(qudits=(1, ))
    op2 = Operation(qudits=(1, 0))
    op3 = Operation(qudits=(1,), cbits=(0,))
    op4 = Operation(cbits=(1, ))
    op5 = Operation(cbits=(0, 1))
    op6 = Operation(qudits=(0, 1), cbits=(0, 1))
    op7 = Operation(qudits=(0, ))
    dag.add_op(op1)
    dag.add_op(op2)
    dag.add_op(op3)
    dag.add_op(op4)
    dag.add_op(op5)
    dag.add_op(op6)
    dag.add_op(op7)

    dag.validate()

    op_order = dag.operation_list()
    assert op_order.index(op_q0_in) < op_order.index(op2)
    assert op_order.index(op_q1_in) < op_order.index(op1) < op_order.index(op2) \
           < op_order.index(op3) < op_order.index(op5) < op_order.index(op6) \
           < op_order.index(op7) < op_order.index(op_q0_out)
    assert op_order.index(op6) < op_order.index(op_q1_out)
    assert op_order.index(op6) < op_order.index(op_c0_out)
    assert op_order.index(op6) < op_order.index(op_c1_out)
    assert op_order.index(op_c0_in) < op_order.index(op3)
    assert op_order.index(op_c1_in) < op_order.index(op4) < op_order.index(op5)


def test_validate_correct():
    """
    Intentionally breaks DAG circuit structure to see that we can detect errors
    """
    dag = CircuitDAG(2, 2)

    op1 = Operation(qudits=(1,))
    op2 = Operation(qudits=(1, 0))
    op3 = Operation(qudits=(1,), cbits=(0,))
    dag.add_op(op1)
    dag.add_op(op2)
    dag.add_op(op3)

    # sabotage graph -- note that we should not directly manipulate the DiGraph except in tests
    dag.DAG.remove_edge(3, 'c0_out')
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
        print(f'gate {i}')
        q_register_num_max = random.randint(1, q)
        c_register_num_max = random.randint(0, c)
        print(q_register_num_max)
        print(c_register_num_max)

        q_registers = tuple(set([random.randint(0, q - 1) for _ in range(q_register_num_max)]))
        c_registers = tuple(set([random.randint(0, c - 1) for _ in range(c_register_num_max)]))
        print(q_registers)
        print(c_registers)
        dag.add_op(Operation(qudits=q_registers, cbits=c_registers))

    dag.validate()
