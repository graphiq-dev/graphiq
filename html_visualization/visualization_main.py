from draw import Painter
from openqasm_parser import OpenQASMParser

openqasm_str = """
OPENQASM 2.0;

qreg q[4];
qreg p[10];
qreg e[2];
creg c[4];
"""

painter = Painter()

painter.add_register(reg_name="p", size=4)
painter.add_register(reg_name="e", size=1)
painter.add_register(reg_name="c", size=4, reg_type="creg")

painter.add_gate(gate_name="H", qargs=["e0"])
painter.add_gate(gate_name="H", qargs=["e0"])
painter.add_gate(gate_name="RX", qargs=["p0"], params={"theta": "pi/2", "phi": "pi/4"})

painter.build_visualization_info()
print(painter.col_width)

for gate in painter.gates:
    print(gate)

