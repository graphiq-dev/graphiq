import requests
import json
from src.utils.draw import Painter

painter = Painter()

painter.add_register(reg_name="p", size=4)
painter.add_register(reg_name="e", size=1)
painter.add_register(reg_name="c", size=4, reg_type="creg")

painter.add_gate(gate_name="H", qargs=["e0"])
painter.add_gate(gate_name="RX", qargs=["p0"], params={"theta": "pi/2", "phi": "pi/4"})
painter.add_gate(gate_name="H", qargs=["p0"])

visualization_info = painter.build_visualization_info()
url = 'http://127.0.0.1:5000/circuit_data'

data = json.dumps(visualization_info, indent=3)
x = requests.post(url, data=data)

print(x.status_code)

