import pytest
import random

from src.circuit import CircuitDAG
from src.ops import OperationBase, CNOT, SigmaX


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
    op_q0_in = dag.dag.nodes['q0-0_in']['op']
    op_q1_in = dag.dag.nodes['q1-0_in']['op']
    op_q0_out = dag.dag.nodes['q0-0_out']['op']
    op_q1_out = dag.dag.nodes['q1-0_out']['op']

    op1 = OperationBase(q_registers=(0,))
    op2 = OperationBase(q_registers=(0,))
    op3 = OperationBase(q_registers=(0,))
    op4 = OperationBase(q_registers=(1,))
    dag.add(op1)
    dag.add(op2)
    dag.add(op3)
    dag.add(op4)

    # make sure all sources are Input operations, all sinks are Output operations
    dag.validate()

    # Retrieve the operators which are actually placed in the graph
    # (not same as those you place in, if you use register inputs)
    op1 = dag.dag.nodes[1]['op']
    op2 = dag.dag.nodes[2]['op']
    op3 = dag.dag.nodes[3]['op']
    op4 = dag.dag.nodes[4]['op']

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

    op_q0_in = dag.dag.nodes['q0-0_in']['op']
    op_q1_in = dag.dag.nodes['q1-0_in']['op']
    op_c0_in = dag.dag.nodes['c0-0_in']['op']
    op_c1_in = dag.dag.nodes['c1-0_in']['op']
    op_q0_out = dag.dag.nodes['q0-0_out']['op']
    op_q1_out = dag.dag.nodes['q1-0_out']['op']
    op_c0_out = dag.dag.nodes['c0-0_out']['op']
    op_c1_out = dag.dag.nodes['c1-0_out']['op']

    op1 = OperationBase(q_registers=(1,))
    op2 = OperationBase(q_registers=(1, 0))
    op3 = OperationBase(q_registers=(1,), c_registers=(0,))
    op4 = OperationBase(c_registers=(1,))
    op5 = OperationBase(c_registers=(0, 1))
    op6 = OperationBase(q_registers=(0, 1), c_registers=(0, 1))
    op7 = OperationBase(q_registers=(0,))
    dag.add(op1)
    dag.add(op2)
    dag.add(op3)
    dag.add(op4)
    dag.add(op5)
    dag.add(op6)
    dag.add(op7)

    dag.validate()

    op1 = dag.dag.nodes[1]['op']
    op2 = dag.dag.nodes[2]['op']
    op3 = dag.dag.nodes[3]['op']
    op4 = dag.dag.nodes[4]['op']
    op5 = dag.dag.nodes[5]['op']
    op6 = dag.dag.nodes[6]['op']
    op7 = dag.dag.nodes[7]['op']

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

    op1 = OperationBase(q_registers=(1,))
    op2 = OperationBase(q_registers=(1, 0))
    op3 = OperationBase(q_registers=(1,), c_registers=(0,))
    dag.add(op1)
    dag.add(op2)
    dag.add(op3)

    # sabotage graph -- note that we should not directly manipulate the DiGraph except in tests
    dag.dag.remove_edge(3, 'c0-0_out')
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

        dag.add(OperationBase(q_registers=q_registers, c_registers=c_registers))

    dag.validate()


def test_dynamic_registers_1():
    """
    Check with single-register gates only
    """
    dag = CircuitDAG(1, 0)
    op1 = OperationBase(q_registers=(1,))
    op2 = OperationBase(q_registers=(2,))
    op3 = OperationBase(c_registers=(0,))
    dag.add(op1)
    dag.validate()
    dag.add(op2)
    dag.validate()
    dag.add(op3)
    dag.validate()

    # retrieve internal information--this should not be done other than for testing
    op_q0_in = dag.dag.nodes['q0-0_in']['op']
    op_q1_in = dag.dag.nodes['q1-0_in']['op']
    op_q2_in = dag.dag.nodes['q2-0_in']['op']
    op_c5_in = dag.dag.nodes['c0-0_in']['op']

    op_q0_out = dag.dag.nodes['q0-0_out']['op']
    op_q1_out = dag.dag.nodes['q1-0_out']['op']
    op_q2_out = dag.dag.nodes['q2-0_out']['op']
    op_c5_out = dag.dag.nodes['c0-0_out']['op']

    op1 = dag.dag.nodes[1]['op']
    op2 = dag.dag.nodes[2]['op']
    op3 = dag.dag.nodes[3]['op']

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


def test_continuous_indices_registers():
    """
    Check with single-register gates only
    """
    dag = CircuitDAG(1, 0)
    op1 = OperationBase(q_registers=(1,))
    op2 = OperationBase(q_registers=(2,))
    op3 = OperationBase(c_registers=(5,))
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
    dag = CircuitDAG(1, 0)
    op1 = OperationBase(q_registers=(1, 2))
    op2 = OperationBase(q_registers=(2,))
    op3 = OperationBase(q_registers=(0,), c_registers=(1, 0))
    dag.add(op1)
    dag.add(op2)
    dag.add(op3)
    dag.validate()

    # retrieve internal information--this should not be done other than for testing
    op_q0_in = dag.dag.nodes['q0-0_in']['op']
    op_q1_in = dag.dag.nodes['q1-0_in']['op']
    op_q2_in = dag.dag.nodes['q2-0_in']['op']
    op_c0_in = dag.dag.nodes['c0-0_in']['op']
    op_c5_in = dag.dag.nodes['c0-0_in']['op']

    op_q0_out = dag.dag.nodes['q0-0_out']['op']
    op_q1_out = dag.dag.nodes['q1-0_out']['op']
    op_q2_out = dag.dag.nodes['q2-0_out']['op']
    op_c0_out = dag.dag.nodes['c0-0_out']['op']
    op_c5_out = dag.dag.nodes['c0-0_out']['op']

    op1 = dag.dag.nodes[1]['op']
    op2 = dag.dag.nodes[2]['op']
    op3 = dag.dag.nodes[3]['op']

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
    dag = CircuitDAG(150, 120)
    dag.validate()
    for i in range(200):  # we'll apply 200 random gates
        q_register_num = random.randint(1, 5)
        c_register_num = random.randint(1, 5)

        q_registers = tuple(set([random.randint(0, 150) for _ in range(q_register_num)]))
        c_registers = tuple(set([random.randint(0, 120) for _ in range(c_register_num)]))
        dag.add(OperationBase(q_registers=q_registers, c_registers=c_registers))

    dag.validate()

# Test register implementations


def test_add_register_1():
    """
    Test the fact we can't add registers of size 0
    """
    dag = CircuitDAG(0, 0)
    dag.validate()
    with pytest.raises(ValueError):
        dag.add_quantum_register(0)
    dag.validate()


def test_add_register_2():
    dag = CircuitDAG(2, 2)
    dag.validate()
    dag.add_quantum_register()
    dag.add(OperationBase(q_registers=(0,)))
    dag.validate()
    dag.add(OperationBase(q_registers=(1, 0)))
    dag.add_classical_register(2)
    dag.validate()

    assert dag.n_quantum == 3
    assert dag.n_classical == 3
    assert dag.dag.number_of_nodes() == 16
    assert dag.dag.number_of_edges() == 10


@pytest.mark.parametrize("is_quantum", [True, False])
def test_expand_register_1(is_quantum):
    """
    Test that you get an error when you try to expand a register that
    does not exist
    """
    dag = CircuitDAG(1, 1)
    dag.validate()
    with pytest.raises(IndexError):
        if is_quantum:
            dag.expand_quantum_register(1, 3)
        else:
            dag.expand_classical_register(1, 3)

    dag.validate()


@pytest.mark.parametrize("is_quantum", [True, False])
def test_expand_register_2(is_quantum):
    """
    Test that you get an error when you try to shrink or not expand a register
    """
    dag = CircuitDAG(1, 1)
    dag.validate()
    with pytest.raises(ValueError):
        if is_quantum:
            dag.expand_quantum_register(0, 0)
        else:
            dag.expand_classical_register(0, 0)

    with pytest.raises(ValueError):
        if is_quantum:
            dag.expand_quantum_register(0, 1)
        else:
            dag.expand_classical_register(0, 1)

    if is_quantum:
        dag.expand_quantum_register(0, 5)
    else:
        dag.expand_classical_register(0, 5)

    assert dag.n_quantum == 1
    assert dag.n_classical == 1
    if is_quantum:
        assert dag.q_registers[0] == 5
        assert dag.c_registers[0] == 1
    else:
        assert dag.c_registers[0] == 5
        assert dag.q_registers[0] == 1

    with pytest.raises(ValueError):
        if is_quantum:
            dag.expand_quantum_register(0, 3)
        else:
            dag.expand_classical_register(0, 3)

    with pytest.raises(ValueError):
        if is_quantum:
            dag.expand_quantum_register(0, 5)
        else:
            dag.expand_classical_register(0, 5)

    if is_quantum:
        dag.expand_quantum_register(0, 7)
        assert dag.q_registers[0] == 7
        assert dag.c_registers[0] == 1
    else:
        dag.expand_classical_register(0, 7)
        assert dag.c_registers[0] == 7
        assert dag.q_registers[0] == 1

    assert dag.n_quantum == 1
    assert dag.n_classical == 1
    assert dag.dag.number_of_nodes() == 16
    assert dag.dag.number_of_edges() == 8

    dag.validate()


def test_nonconsecutive_register_qudit_indexing_1():
    """
    Verify that the quantum registers report an error when non-consecutive qudit indexing is used
    """
    dag = CircuitDAG(2, 0)
    dag.expand_quantum_register(0, 2)
    dag.add(OperationBase(q_registers=((0, 2),)))  # this should work because we've expanded the quantum register 0

    with pytest.raises(ValueError):
        dag.add(OperationBase(q_registers=((1, 2),)))  # this should not work, because we do not have a qubit 1 in reg 1

    dag.validate()


def test_nonconsecutive_register_cbit_indexing_2():
    """
    Verify that the quantum registers report an error when non-consecutive qudit indexing is used
    """
    dag = CircuitDAG(2, 2)
    dag.expand_classical_register(0, 2)
    dag.add(OperationBase(c_registers=((0, 2),)))  # this should work because we've expanded the classical register 0

    with pytest.raises(ValueError):
        dag.add(OperationBase(c_registers=((1, 2),)))  # this should not work, because we do not have a cbit 1 in reg 1

    dag.validate()


@pytest.mark.parametrize("is_quantum", [True, False])
def test_dynamic_register_expansion_1(is_quantum):
    """
    Test that we get an error when non-continuous register numbers are provided (quantum and classical)
    """
    dag = CircuitDAG()

    with pytest.raises(ValueError):
        if is_quantum:
            dag.add(OperationBase(q_registers=(1,)))
        else:
            dag.add(OperationBase(c_registers=(1,)))

    if is_quantum:
        dag.add(OperationBase(q_registers=(0, 0)))
        dag.add(OperationBase(q_registers=((1, 0), (1, 1))))
    else:
        dag.add(OperationBase(c_registers=(0, 0)))
        dag.add(OperationBase(c_registers=((1, 0), (1, 1))))

    with pytest.raises(ValueError):
        if is_quantum:
            dag.add(OperationBase(q_registers=(3, )))
        else:
            dag.add(OperationBase(c_registers=(3, )))

    with pytest.raises(ValueError):
        if is_quantum:
            dag.add(OperationBase(q_registers=((3, 0),)))
        else:
            dag.add(OperationBase(c_registers=((3, 0),)))

    dag.validate()


@pytest.mark.parametrize("is_quantum", [True, False])
def test_dynamic_register_expansion_2(is_quantum):
    """
    Test that we get an error when non-continuous qudit/cbit numbers are provided (quantum and classical)
    """
    dag = CircuitDAG(2, 2)
    if is_quantum:
        dag.add(OperationBase(q_registers=((2, 0),)))
    else:
        dag.add(OperationBase(c_registers=((2, 0),)))

    with pytest.raises(ValueError):
        if is_quantum:
            dag.add(OperationBase(q_registers=((2, 2),)))
        else:
            dag.add(OperationBase(c_registers=((2, 2),)))

    dag.validate()


def test_dynamic_register_expansion_3():
    """
    Test that it works alright with provided numbers. Confirm that topological order / number of nodes/edges are as
    expected
    """
    dag = CircuitDAG(1, 0)
    dag.add(OperationBase(q_registers=(0,), c_registers=((0, 0), (0, 1))))
    dag.add(OperationBase(q_registers=(1,)))

    op_q0_0_in = dag.dag.nodes['q0-0_in']['op']
    op_q1_0_in = dag.dag.nodes['q1-0_in']['op']
    op_c0_0_in = dag.dag.nodes['c0-0_in']['op']
    op_c0_1_in = dag.dag.nodes['c0-1_in']['op']

    op_q0_0_out = dag.dag.nodes['q0-0_out']['op']
    op_q1_0_out = dag.dag.nodes['q1-0_out']['op']
    op_c0_0_out = dag.dag.nodes['c0-0_out']['op']
    op_c0_1_out = dag.dag.nodes['c0-1_out']['op']

    op1 = dag.dag.nodes[1]['op']
    op2 = dag.dag.nodes[2]['op']

    # check topological order
    op_order = dag.sequence()
    assert op_order.index(op_q0_0_in) < op_order.index(op1) < op_order.index(op_q0_0_out)
    assert op_order.index(op_q0_0_in) < op_order.index(op1) < op_order.index(op_c0_0_out)
    assert op_order.index(op_q0_0_in) < op_order.index(op1) < op_order.index(op_c0_1_out)
    assert op_order.index(op_c0_0_in) < op_order.index(op1) < op_order.index(op_q0_0_out)
    assert op_order.index(op_c0_0_in) < op_order.index(op1) < op_order.index(op_c0_0_out)
    assert op_order.index(op_c0_0_in) < op_order.index(op1) < op_order.index(op_c0_1_out)
    assert op_order.index(op_c0_1_in) < op_order.index(op1) < op_order.index(op_q0_0_out)
    assert op_order.index(op_c0_1_in) < op_order.index(op1) < op_order.index(op_c0_0_out)
    assert op_order.index(op_c0_1_in) < op_order.index(op1) < op_order.index(op_c0_1_out)
    assert op_order.index(op_q1_0_in) < op_order.index(op2) < op_order.index(op_q1_0_out)

    assert dag.n_quantum == 2
    assert dag.n_classical == 1
    assert dag.dag.number_of_nodes() == 10
    assert dag.dag.number_of_edges() == 8

    dag.validate()


def test_dynamic_register_expansion_4():
    """
    Test that it works with our next_cbit, next_qubit functions. Confirm that topological order / number of nodes/edges
    are as expected
    """
    dag = CircuitDAG(0, 0)
    next_reg = dag.n_classical
    dag.add(OperationBase(q_registers=((next_reg, 0),)))
    dag.add(OperationBase(q_registers=((next_reg, dag.next_qubit(next_reg)),)))

    op1 = dag.dag.nodes[1]['op']
    op2 = dag.dag.nodes[2]['op']

    assert op1.q_registers == ((0, 0),)
    assert op2.q_registers == ((0, 1),)


def test_dynamic_register_expansion_5():
    """
    Confirm that it does not let you query next_qubit for a register that does not exist
    """
    dag = CircuitDAG(0, 0)
    next_reg = dag.n_classical
    dag.add(OperationBase(q_registers=((next_reg, 0),)))

    with pytest.raises(IndexError):
        dag.next_cbit(0)

    with pytest.raises(IndexError):
        dag.next_qubit(1)


def test_dynamic_register_usage_0():
    """
    Confirm that we get an error when trying to apply register operations between 2 different sized registers
    """
    dag = CircuitDAG(2, 0)
    dag.expand_quantum_register(0, 3)
    dag.expand_quantum_register(1, 2)

    with pytest.raises(AssertionError):
        dag.add(CNOT(control=0, target=1))


def test_dynamic_register_usage_1():
    """
    Confirm that we get the correct graph when applying a register-wide single-qudit/cbit gate
    """
    dag = CircuitDAG(1, 0)
    dag.expand_quantum_register(0, 3)
    dag.add(SigmaX(register=0))
    dag.validate()

    op_order = dag.sequence()

    op_q0_0_in = dag.dag.nodes['q0-0_in']['op']
    op_q0_1_in = dag.dag.nodes['q0-1_in']['op']
    op_q0_2_in = dag.dag.nodes['q0-2_in']['op']
    op_q0_0_out = dag.dag.nodes['q0-0_out']['op']
    op_q0_1_out = dag.dag.nodes['q0-1_out']['op']
    op_q0_2_out = dag.dag.nodes['q0-2_out']['op']

    op1 = dag.dag.nodes[1]['op']
    op2 = dag.dag.nodes[2]['op']
    op3 = dag.dag.nodes[3]['op']

    assert op_order.index(op_q0_0_in) < op_order.index(op1) < op_order.index(op_q0_0_out)
    assert op_order.index(op_q0_1_in) < op_order.index(op2) < op_order.index(op_q0_1_out)
    assert op_order.index(op_q0_2_in) < op_order.index(op3) < op_order.index(op_q0_2_out)

    assert dag.n_quantum == 1
    assert dag.n_classical == 0
    assert dag.dag.number_of_nodes() == 9
    assert dag.dag.number_of_edges() == 6


def test_dynamic_register_usage_2():
    """
    Confirm that we can apply 2 qubit gates between 2 registers of the same size
    """
    dag = CircuitDAG(2, 0)
    dag.expand_quantum_register(0, 2)
    dag.expand_quantum_register(1, 2)
    dag.add(CNOT(control=0, target=1))
    dag.validate()

    op_q0_0_in = dag.dag.nodes['q0-0_in']['op']
    op_q0_1_in = dag.dag.nodes['q0-1_in']['op']
    op_q0_0_out = dag.dag.nodes['q0-0_out']['op']
    op_q0_1_out = dag.dag.nodes['q0-1_out']['op']

    op_q1_0_in = dag.dag.nodes['q1-0_in']['op']
    op_q1_1_in = dag.dag.nodes['q1-1_in']['op']
    op_q1_0_out = dag.dag.nodes['q1-0_out']['op']
    op_q1_1_out = dag.dag.nodes['q1-1_out']['op']

    op1 = dag.dag.nodes[1]['op']
    op2 = dag.dag.nodes[2]['op']

    op_order = dag.sequence()
    assert op_order.index(op_q0_0_in) < op_order.index(op1) < op_order.index(op_q0_0_out)
    assert op_order.index(op_q0_1_in) < op_order.index(op2) < op_order.index(op_q0_1_out)
    assert op_order.index(op_q1_0_in) < op_order.index(op1) < op_order.index(op_q1_0_out)
    assert op_order.index(op_q1_1_in) < op_order.index(op2) < op_order.index(op_q1_1_out)

    assert dag.n_quantum == 2
    assert dag.n_classical == 0
    assert dag.dag.number_of_nodes() == 10
    assert dag.dag.number_of_edges() == 8


def test_dynamic_register_usage_3():
    """
    Confirm that a reg-qubit specific gate works correctly
    """
    dag = CircuitDAG(2, 0)
    dag.expand_quantum_register(0, 2)
    dag.add(SigmaX(register=(0, 0)))
    dag.add(CNOT(control=(0, 0), target=(0, 1)))

    op_q0_0_in = dag.dag.nodes['q0-0_in']['op']
    op_q0_1_in = dag.dag.nodes['q0-1_in']['op']
    op_q0_0_out = dag.dag.nodes['q0-0_out']['op']
    op_q0_1_out = dag.dag.nodes['q0-1_out']['op']
    op_q1_0_in = dag.dag.nodes['q1-0_in']['op']
    op_q1_0_out = dag.dag.nodes['q1-0_out']['op']

    op1 = dag.dag.nodes[1]['op']
    op2 = dag.dag.nodes[2]['op']

    dag.validate()
    op_order = dag.sequence()
    assert op_order.index(op_q0_0_in) < op_order.index(op1) < op_order.index(op2) < op_order.index(op_q0_0_out)
    assert op_order.index(op_q0_1_in) < op_order.index(op2) < op_order.index(op_q0_1_out)
    assert op_order.index(op_q1_0_in) < op_order.index(op_q1_0_out)

    assert dag.n_quantum == 2
    assert dag.n_classical == 0
    assert dag.dag.number_of_nodes() == 8
    assert dag.dag.number_of_edges() == 6


def test_dynamic_register_usage_4():
    """
    Confirm that two qubit gate of form q/c_register=(a, (b, c)) works correctly
    """
    dag = CircuitDAG(2, 0)
    dag.expand_quantum_register(0, 2)
    dag.add(CNOT(control=0, target=(1, 1)))

    dag.validate()

    op_q0_0_in = dag.dag.nodes['q0-0_in']['op']
    op_q0_0_out = dag.dag.nodes['q0-0_out']['op']
    op_q0_1_in = dag.dag.nodes['q0-1_in']['op']
    op_q0_1_out = dag.dag.nodes['q0-1_out']['op']
    op_q1_0_in = dag.dag.nodes['q1-0_in']['op']
    op_q1_0_out = dag.dag.nodes['q1-0_out']['op']
    op_q1_1_in = dag.dag.nodes['q1-1_in']['op']
    op_q1_1_out = dag.dag.nodes['q1-1_out']['op']

    op1 = dag.dag.nodes[1]['op']
    op2 = dag.dag.nodes[2]['op']

    op_order = dag.sequence()

    assert op_order.index(op_q0_0_in) < op_order.index(op1) < op_order.index(op_q0_0_out)
    assert op_order.index(op_q0_1_in) < op_order.index(op2) < op_order.index(op_q0_1_out)
    assert op_order.index(op_q1_0_in) < op_order.index(op_q1_0_out)
    assert op_order.index(op_q1_1_in) < op_order.index(op1) < op_order.index(op_q1_1_out)
    assert op_order.index(op_q1_1_in) < op_order.index(op2) < op_order.index(op_q1_1_out)

    assert dag.n_quantum == 2
    assert dag.n_classical == 0
    assert dag.dag.number_of_nodes() == 10
    assert dag.dag.number_of_edges() == 8


def test_dynamic_register_usage_5():
    """
    Confirm that two qubit gate of form q/c_register=((a, b), c) works correctly
    """
    dag = CircuitDAG(1, 1)
    dag.add(OperationBase(q_registers=(0, ), c_registers=((0, 0),)))

    dag.validate()

    op_q0_0_in = dag.dag.nodes['q0-0_in']['op']
    op_q0_0_out = dag.dag.nodes['q0-0_out']['op']
    op_c0_0_in = dag.dag.nodes['c0-0_in']['op']
    op_c0_0_out = dag.dag.nodes['c0-0_out']['op']
    op1 = dag.dag.nodes[1]['op']

    op_order = dag.sequence()

    assert op_order.index(op_q0_0_in) < op_order.index(op1) < op_order.index(op_q0_0_out)
    assert op_order.index(op_c0_0_in) < op_order.index(op1) < op_order.index(op_c0_0_out)

    assert dag.n_quantum == 1
    assert dag.n_classical == 1
    assert dag.dag.number_of_nodes() == 5
    assert dag.dag.number_of_edges() == 4


def test_register_qubit_assignment_1():
    dag = CircuitDAG(1, 1)
    dag.add(SigmaX(register=0))
    dag.add(SigmaX(register=0))
    op_order = dag.sequence()

    op_q0_0_in = dag.dag.nodes['q0-0_in']['op']
    op_c0_0_in = dag.dag.nodes['c0-0_in']['op']
    op_q0_0_out = dag.dag.nodes['q0-0_out']['op']
    op_c0_0_out = dag.dag.nodes['c0-0_out']['op']

    op1 = dag.dag.nodes[1]['op']
    op2 = dag.dag.nodes[2]['op']

    # check topological order
    assert op_order.index(op_q0_0_in) < op_order.index(op1) < op_order.index(op2) < op_order.index(op_q0_0_out)
    assert op_order.index(op_c0_0_in) < op_order.index(op_c0_0_out)

    # Test that the action id has been assigned to 1
    assert op_q0_0_in.register == (0, 0)
    assert op1.register == (0, 0)
    assert op2.register == (0, 0)
    assert op_q0_0_out.register == (0, 0)
    assert op_c0_0_in.register == (0, 0)
    assert op_c0_0_out.register == (0, 0)


def test_action_id_assignment_2():
    dag = CircuitDAG(2, 1)
    dag.expand_quantum_register(0, 3)
    dag.add(SigmaX(register=0))
    dag.add(CNOT(control=(0, 0), target=(1, 0)))
    dag.add(CNOT(control=0, target=(1, 0)))

    op_order = dag.sequence()

    op_q0_0_in = dag.dag.nodes['q0-0_in']['op']
    op_q0_1_in = dag.dag.nodes['q0-1_in']['op']
    op_q0_2_in = dag.dag.nodes['q0-2_in']['op']
    op_q1_0_in = dag.dag.nodes['q1-0_in']['op']
    op_c0_0_in = dag.dag.nodes['c0-0_in']['op']

    op_q0_0_out = dag.dag.nodes['q0-0_out']['op']
    op_q0_1_out = dag.dag.nodes['q0-1_out']['op']
    op_q0_2_out = dag.dag.nodes['q0-2_out']['op']
    op_q1_0_out = dag.dag.nodes['q1-0_out']['op']
    op_c0_0_out = dag.dag.nodes['c0-0_out']['op']

    ops = [dag.dag.nodes[i]['op'] for i in range(1, 8)]

    # Test that the action ids have been assigned correctly
    assert op_q0_0_in.register == op_q0_0_out.register == (0, 0)
    assert op_q0_1_in.register == op_q0_1_out.register == (0, 1)
    assert op_q0_2_in.register == op_q0_2_out.register == (0, 2)
    assert op_q1_0_in.register == op_q1_0_out.register == (1, 0)
    assert op_c0_0_in.register == op_c0_0_out.register == (0, 0)

    assert ops[0].register == (0, 0)
    assert ops[1].register == (0, 1)
    assert ops[2].register == (0, 2)
    assert ops[3].control == (0, 0)
    assert ops[3].target == (1, 0)
    assert ops[4].control == (0, 0)
    assert ops[4].target == (1, 0)
    assert ops[5].control == (0, 1)
    assert ops[5].target == (1, 0)
    assert ops[6].control == (0, 2)
    assert ops[6].target == (1, 0)
