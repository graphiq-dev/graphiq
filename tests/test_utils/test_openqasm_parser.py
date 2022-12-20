import pytest
from qiskit.qasm import Qasm
from src.ops import *
from src.utils.openqasm_parser import OpenQASMParser
from src.circuit import CircuitDAG
@pytest.mark.parametrize("qreg, expected",
                         [("qreg q[5];", {"index": 5}), ("qreg q[3];", {"index": 3}), ("qreg q[10];", {"index": 10})])
def test_parser_parse_qreg(qreg, expected):
    # create a parser object
    parser = Qasm(data=qreg)

    # parse the qasm code
    res = parser.parse()
    ans = res.children[0]
    print(vars(ans))

    myparser = OpenQASMParser(openqasm_str="")
    assert myparser._parse_qreg(ans) == expected


@pytest.mark.parametrize("creg, expected",
                         [("creg c[2];", {"index": 2}), ("creg c[4];", {"index": 4}), ("creg c[10];", {"index": 10})])
def test_parser_parse_creg(creg, expected):
    # create a parser object
    parser = Qasm(data=creg)

    # parse the qasm code
    res = parser.parse()
    ans = res.children[0]
    print(vars(ans))

    myparser = OpenQASMParser(openqasm_str="")
    assert myparser._parse_creg(ans) == expected


@pytest.mark.parametrize("gate, expected",
                         [("gate h q {U(pi,0,pi) q;}", {'type': 'gate', 'params': [], 'qargs': ['q']}),
                          ("gate rx(theta) q {U(pi/2,theta,pi/2) q;}",
                           {'type': 'gate', 'params': ['theta'], 'qargs': ['q']})])
def test_parser_parse_gate(gate, expected):
    # create a parser object
    parser = Qasm(data=gate)

    # parse the qasm code
    res = parser.parse()
    ans = res.children[0]
    print(vars(ans))

    myparser = OpenQASMParser(openqasm_str="")
    assert myparser._parse_gate(ans) == expected


def test_parser_parse_barrier():
    # create a parser object
    openqasm = """
    qreg q[3];
    
    barrier q[1];
    """
    parser = Qasm(data=openqasm)

    # parse the qasm code
    res = parser.parse()
    ans = res.children[1]
    print(vars(ans))

    expected = {'type': 'barrier', 'qreg': ['q']}

    myparser = OpenQASMParser(openqasm_str="")
    assert myparser._parse_barrier(ans) == expected


def test_parser_parse_measure():
    openqasm = """
    qreg q[3];
    creg c[3];
    
    measure q[1] -> c[1];
    """

    # create a parser object
    parser = Qasm(data=openqasm)

    # parse the qasm code
    res = parser.parse()
    ans1 = res.children[0]
    ans2 = res.children[1]
    print((vars(ans1)))
    print((vars(ans2)))

    expected1 = {'type': 'qreg', 'qreg': {'name': 'q', 'index': 3}, 'creg': {'name': 'q', 'index': 3}}
    expected2 = {'type': 'creg', 'qreg': {'name': 'c', 'index': 3}, 'creg': {'name': 'c', 'index': 3}}

    myparser = OpenQASMParser(openqasm_str="")
    assert myparser._parse_measure(ans1) == expected1
    assert myparser._parse_measure(ans2) == expected2


def test_parser_parse_reset():
    openqasm = """
    qreg q[3];
    reset q[1];
    """

    # create a parser object
    parser = Qasm(data=openqasm)

    # parse the qasm code
    res = parser.parse()
    ans = res.children[1]
    print(vars(ans))

    expected = {'type': 'reset', 'name': 'q', 'index': 1}

    myparser = OpenQASMParser(openqasm_str="")
    assert myparser._parse_reset(ans) == expected


def test_parser_parse_expression():
    pass


def test_parser_parse_custom_unitary():
    circuit = CircuitDAG(n_photon=4, n_classical=4)
    circuit.add(operation=Hadamard(reg_type="p", register=1))
    openqasm = circuit.to_openqasm()

    # create a parser object
    parser = Qasm(data=openqasm)

    # parse the qasm code
    res = parser.parse()
    ans = res.children[-1]
    print(vars(ans))

    expected = {'type': 'custom_unitary', 'name': 'h', 'params': {}, 'qargs': [('p1', 0)]}

    myparser = OpenQASMParser(openqasm_str=openqasm)
    myparser.parse()
    assert myparser._parse_custom_unitary(ans) == expected


def test_parser_parse_cnot():
    # create a parser object
    openqasm = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[3];
    creg c[3];
   
    CX q[1], q[2];
    
    """
    parser = Qasm(data=openqasm)

    # parse the qasm code
    res = parser.parse()
    ans = res.children[-1]
    print(vars(ans))

    expected = {'type': 'cnot', 'control': {'name': 'q', 'index': 1}, 'target': {'name': 'q', 'index': 2}}

    myparser = OpenQASMParser(openqasm_str="")
    assert myparser._parse_cnot(ans) == expected


def test_parser_parse_if():
    # create a parser object
    openqasm = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[3];
    creg c[3];

    if(c==1) CX q[1], q[2];
    """
    parser = Qasm(data=openqasm)

    # parse the qasm code
    res = parser.parse()
    ans = res.children[-1]
    print(vars(ans))

    expected = {'type': 'if'}

    myparser = OpenQASMParser(openqasm_str="")
    assert myparser._parse_if(ans) == expected
