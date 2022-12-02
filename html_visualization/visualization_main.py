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

# painter.add_gate(gate_name="H", qargs=["e0"])
# painter.add_gate(gate_name="RX", qargs=["e0"], params=["pi/2", "pi/4"])
# painter.add_gate(gate_name="RX", qargs=["e0"], params=["pi/2", "pi/4"])
painter.add_gate(gate_name="H", qargs=["e0"])
painter.add_gate(gate_name="CX", qargs=["e0", "p1"])
painter.add_gate(gate_name="CX", qargs=["e0", "p2"])

info = painter.build_visualization_info()

print(info)

# print(painter.col_width)
# print(painter.build_visualization_info())


