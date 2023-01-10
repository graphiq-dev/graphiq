import json
import requests
from src.utils.openqasm_parser import OpenQASMParser

standard_gate_width_mapping = {
    "CX": 40,
    "CZ": 40,
}


class Columns:
    def __init__(self, col_num: int, size: int = 1):
        self.size = size
        self.columns = []

        for i in range(col_num):
            self.columns.append([0] * size)

        self.col_width = [0] * col_num

    def add_new_column(self):
        new_column = [0] * self.size
        self.columns.append(new_column)
        self.col_width.append(0)

    def update_col_width(self, index, new_width):
        self.col_width[index] = max(self.col_width[index], new_width)

    def expand_cols(self, size: int = 1):
        if not self.columns:
            self.columns.append([0] * size)
            self.col_width.append(0)
        else:
            for col in self.columns:
                col.extend([0] * size)
        self.size += size

    def set_all_col_element(self, index: int, value: int = -1):
        if not isinstance(index, int):
            raise ValueError("Index parameter must be an integer")
        for i in reversed(range(len(self.columns))):
            if not self.columns[i][index]:
                self.columns[i][index] = value
        return

    def find_and_add_to_empty_col(self, from_index: int, to_index: int, value: int = 1):
        gate_col = 0

        for i, v in enumerate(self.columns):
            gate_pos = v[from_index:to_index]

            if i == len(self.columns) - 1 and not all(pos == 0 for pos in gate_pos):
                self.add_new_column()
                self.columns[i + 1][from_index:to_index] = [value] * len(gate_pos)
                gate_col = i + 1
                break
            if all(pos == 0 for pos in gate_pos):
                self.columns[i][from_index:to_index] = [value] * len(gate_pos)
                gate_col = i
                break

        to_index = to_index - 1
        if to_index != from_index:
            for i in reversed(range(gate_col)):
                if not self.columns[i][from_index]:
                    self.columns[i][from_index] = -1
                if not self.columns[i][to_index]:
                    self.columns[i][to_index] = -1
        return gate_col


class Painter:
    def __init__(self, gate_mapping=None):
        self._columns = Columns(col_num=0)
        self.next_reg_position = 50

        self.registers_position = {
            "qreg": {},
            "creg": {},
        }
        self.registers_mapping = []
        self.ops = []

        if not gate_mapping:
            self.gate_mapping = standard_gate_width_mapping
        else:
            self.gate_mapping = gate_mapping

    @staticmethod
    def to_reg_label(reg_name, size):
        return str(reg_name) + "[" + str(size) + "]"

    def _calculate_gate_width(self, gate_name, params):
        params_str = ""
        if params:
            params_str += "("
            for i in params:
                params_str += f"{params[i]}, "
            params_str = params_str[:-2]
            params_str += ")"
        width = (
            self.gate_mapping[gate_name]
            if gate_name.upper() in self.gate_mapping
            else max(40, 15 * 2 + len(gate_name) * 10, 15 * 2 + len(params_str) * 5)
        )
        return width

    def add_register(self, reg_name, size, reg_type="qreg"):
        if reg_type != "qreg" and reg_type != "creg":
            raise ValueError("Register type must be creg or qreg")

        # map register position to the correct coordinates
        if reg_type == "qreg":
            for i in range(size):
                self.registers_position[reg_type][
                    f"{reg_name}[{i}]"
                ] = self.next_reg_position
                self.next_reg_position += 50
                self.registers_mapping.append(f"{reg_name}[{i}]")
            self._columns.expand_cols(size=size)
        else:
            self.registers_position[reg_type][
                f"{reg_name}[{size}]"
            ] = self.next_reg_position
            self.next_reg_position += 50
            self.registers_mapping.append(f"{reg_name}[{size}]")
            self._columns.expand_cols(size=1)

    # TODO: Fix and enable multi controls gate, right now multi-controls gate cause some weird drawing
    def add_gate(
        self, gate_name: str, qargs: list, params: dict = None, controls: list = None
    ):
        if len(qargs) > 1:
            raise ValueError("Gate that act on multi-qargs is not supported yet.")
        # calculate which registers the gate will be on
        reg_pos = [self.registers_mapping.index(arg) for arg in qargs]
        control_pos = (
            [self.registers_mapping.index(arg) for arg in controls] if controls else []
        )
        from_reg = min(reg_pos + control_pos)
        to_reg = max(reg_pos + control_pos) + 1
        gate_col = self._columns.find_and_add_to_empty_col(from_reg, to_reg)

        # calculate the col width
        self._columns.update_col_width(
            index=gate_col, new_width=self._calculate_gate_width(gate_name, params)
        )

        # gate info
        gate_info = {
            "type": "gate",
            "gate_name": gate_name,
            "params": {} if params is None else params,
            "qargs": qargs,
            "controls": [] if controls is None else controls,
            "col": gate_col,
        }
        self.ops.append(gate_info)

        return gate_info

    def add_measurement(self, qreg, creg, cbit=0):
        from_reg = self.registers_mapping.index(qreg)
        to_reg = self.registers_mapping.index(creg) + 1

        measure_col = self._columns.find_and_add_to_empty_col(from_reg, to_reg)
        self._columns.update_col_width(index=measure_col, new_width=40)

        measurement_info = {
            "type": "measure",
            "col": measure_col,
            "qreg": qreg,
            "creg": creg,
            "cbit": cbit,
        }
        self.ops.append(measurement_info)
        return measurement_info

    def add_barriers(self, qreg: list):
        reg_pos = [self.registers_mapping.index(arg) for arg in qreg]
        from_reg = min(reg_pos)
        to_reg = max(reg_pos) + 1

        barriers_col = self._columns.find_and_add_to_empty_col(from_reg, to_reg)
        for i in range(from_reg, to_reg):
            self._columns.set_all_col_element(i)
        self._columns.update_col_width(index=barriers_col, new_width=40)

        for reg in qreg:
            barrier_info = {
                "type": "barrier",
                "col": barriers_col,
                "qreg": reg,
            }
            self.ops.append(barrier_info)

    def add_reset(self, qreg):
        from_reg = self.registers_mapping.index(qreg)
        to_reg = from_reg + 1

        reset_col = self._columns.find_and_add_to_empty_col(from_reg, to_reg)
        self._columns.update_col_width(index=reset_col, new_width=40)
        reset_info = {
            "type": "reset",
            "col": reset_col,
            "qreg": qreg,
        }
        self.ops.append(reset_info)

        return reset_info

    # Classical control gate
    def add_classical_control(
        self, creg, gate_name: str, qargs: list, params: dict = None
    ):
        if len(qargs) > 1:
            raise ValueError(
                "Multiple qubits gate is not supported in classical control right now"
            )

        qreg_pos = [self.registers_mapping.index(arg) for arg in qargs]
        creg_pos = [self.registers_mapping.index(creg)]
        from_reg = min(qreg_pos + creg_pos)
        to_reg = max(qreg_pos + creg_pos) + 1

        # update col width
        gate_col = self._columns.find_and_add_to_empty_col(from_reg, to_reg)
        self._columns.update_col_width(
            index=gate_col, new_width=self._calculate_gate_width(gate_name, params)
        )

        classical_control_info = {
            "type": "if",
            "col": gate_col,
            "creg": creg,
            "gate_info": {
                "gate_name": gate_name,
                "params": {} if params is None else params,
                "qargs": qargs,
                "control": [],
            },
        }
        self.ops.append(classical_control_info)
        return classical_control_info

    def build_visualization_info(self):
        # calculate x_pos for gates and columns
        start = 100
        cols_mid_point = []

        for i, v in enumerate(self._columns.col_width):
            cols_mid_point.append(start + v / 2)
            start += v + 20

        for op in self.ops:
            op["x_pos"] = cols_mid_point[op["col"]]

        visualization_dict = {
            "width": self.ops[-1]["x_pos"] + 100 if self.ops else 1000,
            "registers": self.registers_position,
            "ops": self.ops,
        }
        return visualization_dict

    def draw(self):
        visualization_info = self.build_visualization_info()
        url = "http://127.0.0.1:5000/circuit_data"

        data = json.dumps(visualization_info, indent=3)
        result = requests.post(url, data=data)

        return result

    def load_openqasm_str(self, openqasm_str):
        parser = OpenQASMParser(openqasm_str)
        parser.parse()

        for node in parser.parse():
            if node["type"] == "qreg":
                self.add_register(node["name"], node["size"], "qreg")
            if node["type"] == "creg":
                self.add_register(node["name"], node["size"], "creg")
            if node["type"] == "custom_unitary":
                qargs = [self.to_reg_label(qarg[0], qarg[1]) for qarg in node["qargs"]]
                self.add_gate(str.upper(node["name"]), qargs)
            if node["type"] == "cnot":
                qargs = (
                    node["target"]["name"] + "[" + str(node["target"]["index"]) + "]"
                )
                controls = (
                    node["control"]["name"] + "[" + str(node["control"]["index"]) + "]"
                )
                self.add_gate(gate_name="CX", qargs=[qargs], controls=[controls])
            if node["type"] == "measure":
                creg_size = parser.get_register_size(node["creg"]["name"], "creg")
                if hasattr(node, "index"):
                    qreg = self.to_reg_label(
                        node["qreg"]["name"], node["qreg"]["index"]
                    )
                    creg = self.to_reg_label(node["creg"]["name"], creg_size)
                    self.add_measurement(
                        qreg=qreg, creg=creg, cbit=node["creg"]["index"]
                    )
                else:
                    for i in range(
                        parser.get_register_size(node["qreg"]["name"], "qreg")
                    ):
                        qreg = self.to_reg_label(node["qreg"]["name"], i)
                        creg = self.to_reg_label(node["creg"]["name"], creg_size)
                        self.add_measurement(qreg=qreg, creg=creg, cbit=i)
            if node["type"] == "reset":
                if hasattr(node, "index"):
                    qreg = self.to_reg_label(node["name"], node["index"])
                    self.add_reset(qreg)
                else:
                    for i in range(parser.get_register_size(node["name"], "qreg")):
                        qreg = self.to_reg_label(node["name"], i)
                        self.add_reset(qreg)
            if node["type"] == "barrier":
                qreg_list = []
                for reg in node["qreg"]:
                    for i in range(parser.ast["def"]["qreg"][reg]["size"]):
                        qreg_list.append(f"{reg}[{i}]")
                self.add_barriers(qreg_list)
            if node["type"] == "if":
                creg = (
                    node["creg"]["name"]
                    + "["
                    + str(parser.ast["def"]["creg"][node["creg"]["name"]]["size"])
                    + "]"
                )
                qargs = [
                    self.to_reg_label(
                        node["custom_unitary"]["qargs"][0][0],
                        node["custom_unitary"]["qargs"][0][1],
                    )
                ]
                self.add_classical_control(
                    creg=creg,
                    gate_name=str.upper(node["custom_unitary"]["name"]),
                    params=node["custom_unitary"]["params"],
                    qargs=qargs,
                )
