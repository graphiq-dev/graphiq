from draw import Painter

openqasm_str = """
OPENQASM 2.0;


"""

painter = Painter()

painter.add_register(register=0, reg_type="p")
painter.add_register(register=1, reg_type="p")
painter.add_register(register=2, reg_type="p")
painter.add_register(register=3, reg_type="p")
painter.add_register(register=0, reg_type="e")
painter.add_register(register=0, reg_type="c")

# painter.add_one_qubit_gate(reg_type="e", register=0, gate_name="H")
# painter.add_two_qubit_gate(control_type="e", control=0, target_type="p", target=2)
# painter.add_one_qubit_gate(reg_type="p", register=0, gate_name="H")

print(painter.add_gate(gate_name="H", qargs=["e0"]))
print(painter.add_gate(gate_name="CX", qargs=["e0", "p1"]))
print(painter.add_gate(gate_name="CX", qargs=["e0", "p2"]))
print(painter.registers_col)


