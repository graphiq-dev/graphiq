import math

import pytest
from qiskit.qasm import Qasm

from graphiq.circuit.circuit_dag import CircuitDAG
from graphiq.circuit.ops import *
from graphiq.utils.openqasm_parser import OpenQASMParser


@pytest.mark.parametrize(
    "qreg, expected",
    [
        ("qreg q[5];", {"type": "qreg", "size": 5, "name": "q"}),
        ("qreg p[3];", {"type": "qreg", "size": 3, "name": "p"}),
        ("qreg test[10];", {"type": "qreg", "size": 10, "name": "test"}),
    ],
)
def test_parser_parse_qreg(qreg, expected):
    # create a parser object
    parser = Qasm(data=qreg)

    # parse the qasm code
    res = parser.parse()
    ans = res.children[0]

    myparser = OpenQASMParser(openqasm_str="")
    parse_qreg = myparser.use_parse("qreg")
    assert parse_qreg(ans) == expected


@pytest.mark.parametrize(
    "creg, expected",
    [
        ("creg c[2];", {"type": "creg", "size": 2, "name": "c"}),
        ("creg c[4];", {"type": "creg", "size": 4, "name": "c"}),
        ("creg c[10];", {"type": "creg", "size": 10, "name": "c"}),
    ],
)
def test_parser_parse_creg(creg, expected):
    # create a parser object
    parser = Qasm(data=creg)

    # parse the qasm code
    res = parser.parse()
    ans = res.children[0]

    myparser = OpenQASMParser(openqasm_str="")
    parse_creg = myparser.use_parse("creg")
    assert parse_creg(ans) == expected


@pytest.mark.parametrize(
    "gate, expected",
    [
        (
                "gate h q {U(pi,0,pi) q;}",
                {"type": "gate", "name": "h", "params": [], "qargs": ["q"]},
        ),
        (
                "gate rx(theta) q {U(pi/2,theta,pi/2) q;}",
                {"type": "gate", "name": "rx", "params": ["theta"], "qargs": ["q"]},
        ),
    ],
)
def test_parser_parse_gate(gate, expected):
    # create a parser object
    parser = Qasm(data=gate)

    # parse the qasm code
    res = parser.parse()
    ans = res.children[0]

    myparser = OpenQASMParser(openqasm_str="")
    parse_gate = myparser.use_parse("gate")
    assert parse_gate(ans) == expected


def test_parser_parse_barrier():
    # create a parser object
    openqasm = """
    qreg q[3];

    barrier q[1];
    """
    parser = Qasm(data=openqasm)

    # parse the qasm code
    res = parser.parse()
    ans = res.children[-1]

    expected = {"type": "barrier", "qreg": ["q"]}

    myparser = OpenQASMParser(openqasm_str="")
    parse_barrier = myparser.use_parse("barrier")
    assert parse_barrier(ans) == expected


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
    measure_node = res.children[-1]

    expected_output = {
        "type": "measure",
        "qreg": {"name": "q", "index": 1},
        "creg": {"name": "c", "index": 1},
    }

    myparser = OpenQASMParser(openqasm_str="")
    parse_measure = myparser.use_parse("measure")
    assert parse_measure(measure_node) == expected_output


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

    expected = {"type": "reset", "name": "q", "index": 1}

    myparser = OpenQASMParser(openqasm_str="")
    assert myparser._parse_reset(ans) == expected


def test_parser_parse_custom_unitary():
    circuit = CircuitDAG(n_photon=4, n_classical=4)
    circuit.add(operation=Hadamard(reg_type="p", register=1))
    openqasm = circuit.to_openqasm()

    expected = {
        "type": "custom_unitary",
        "name": "h",
        "params": {},
        "qargs": [("p1", 0)],
    }

    myparser = OpenQASMParser(openqasm_str=openqasm)
    for p in myparser.parse():
        continue
    assert myparser.ast["ops"][-1] == expected


def test_parser_parse_cnot():
    # create a parser object
    openqasm = """
    OPENQASM 2.0;
    qreg q[3];
    creg c[3];

    CX q[1], q[2];
    """
    parser = Qasm(data=openqasm)

    # parse the qasm code
    res = parser.parse()
    ans = res.children[-1]
    print(vars(ans))

    expected = {
        "type": "cnot",
        "control": {"name": "q", "index": 1},
        "target": {"name": "q", "index": 2},
    }

    myparser = OpenQASMParser(openqasm_str="")
    assert myparser._parse_cnot(ans) == expected


def test_parser_parse_if():
    # create a parser object
    openqasm = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[3];
    creg c[3];

    if(c==1) h q[1];
    """
    parser = OpenQASMParser(openqasm_str=openqasm)
    for p in parser.parse():
        continue

    expected = {
        "type": "if",
        "creg": {"name": "c", "value": 1},
        "custom_unitary": {
            "type": "custom_unitary",
            "name": "h",
            "params": {},
            "qargs": [("q", 1)],
        },
    }
    assert parser.ast["ops"][-1] == expected


def test_get_reg_size():
    openqasm = """
    OPENQASM 2.0;
    
    qreg p[4];
    creg c[4];
    """

    parser = OpenQASMParser(openqasm)
    for p in parser.parse():
        continue

    assert parser.get_register_size("p", "qreg") == 4
    assert parser.get_register_size("c", "creg") == 4


def test_custom_unitary_with_params():
    openqasm_str = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[2];

    u3(pi/2, pi/2, pi/2) q[0];
    """

    parser = OpenQASMParser(openqasm_str)
    for p in parser.parse():
        continue

    assert parser.ast["ops"][-1] == {
        "type": "custom_unitary",
        "name": "u3",
        "params": {
            "theta": {"operator": "/", "real": math.pi, "int": 2},
            "phi": {"operator": "/", "real": math.pi, "int": 2},
            "lambda": {"operator": "/", "real": math.pi, "int": 2},
        },
        "qargs": [("q", 0)],
    }
