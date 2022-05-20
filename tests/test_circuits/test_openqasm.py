import pytest
import src.ops as ops
import src.libraries.openqasm_lib as oq_lib


def check_openqasm_equivalency(s1, s2):
    assert "".join(s1.split()) == "".join(s2.split()), f"Strings don't match. S1 is: \n{s1}, \n\n, S2 is: \n{s2}"
    # we remove all white spaces, since openqasm does not consider white spaces


def test_empty_circuit_1(dag):
    openqasm = dag.to_openqasm()
    check_openqasm_equivalency(openqasm, oq_lib.openqasm_header())


def test_gateless_circuit_1(dag):
    dag.add_quantum_register(size=1)
    dag.add_quantum_register(size=1)
    dag.add_classical_register(size=1)
    openqasm = dag.to_openqasm()
    expected = oq_lib.openqasm_header() + oq_lib.register_initialization_string([1, 1], [1])
    check_openqasm_equivalency(openqasm, expected)


def test_gateless_circuit_2(dag):
    dag.add_quantum_register(size=3)
    dag.add_quantum_register(size=1)
    dag.add_classical_register(size=5)
    dag.add_classical_register(size=1)
    dag.add_classical_register(size=2)

    openqasm = dag.to_openqasm()
    expected = oq_lib.openqasm_header() + oq_lib.register_initialization_string([3, 1], [5, 1, 2])
    check_openqasm_equivalency(openqasm, expected)


def test_general_circuit_1(dag):
    dag.add_quantum_register(size=3)
    dag.add_quantum_register(size=3)
    dag.add_classical_register(size=1)
    dag.add(ops.CNOT(control=1, target=0))
    dag.add(ops.SigmaX(register=(0, 0)))
    dag.validate()

    openqasm = dag.to_openqasm()
    expected = 'OpenQASM 3.0;\n import "stdgates.inc";\n qubit[3] q0;\n qubit[3] q1;\n' + \
               'bit[3] c0;\n bit[3] c1;\n cx q1[0], q0[0];\n cx q1[1], q0[1];'
    check_openqasm_equivalency(openqasm, expected)
