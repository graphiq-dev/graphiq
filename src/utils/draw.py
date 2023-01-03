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
        self.columns = [[0] * size] * col_num
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
                col.append(0 * size)
        self.size += size

    def set_all_col_element(self, index: int, value: int = -1):
        if not isinstance(index, int):
            raise ValueError("Index parameter must be an integer")
        for i in reversed(range(len(self.columns) - 1)):
            if not self.columns[i][index]:
                self.columns[i][index] = value
        return

    def find_and_add_to_empty_col(self, from_index: int, to_index: int, value: int = 1):
        gate_col = 0

        for i, v in enumerate(self.columns):
            gate_pos = v[from_index:to_index]

            if i == len(self.columns)-1 and not all(pos == 0 for pos in gate_pos):
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
            for i in reversed(range(len(self.columns) - 1)):
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

        self.gates = []
        self.measurements = []
        self.barriers = []
        self.resets = []
        self.classical_controls = []

        if not gate_mapping:
            self.gate_mapping = standard_gate_width_mapping
        else:
            self.gate_mapping = gate_mapping

    @staticmethod
    def convert_qargs_tuple_to_str(qarg):
        return str(qarg[0]) + str(qarg[1])

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
        reg_pos = {}
        if reg_type == "qreg":
            for i in range(size):
                curr_reg_pos = self.next_reg_position
                self.next_reg_position += 50
                reg_pos[f"{reg_name}{i}"] = self.registers_position[reg_type][
                    f"{reg_name}{i}"
                ] = curr_reg_pos
                self.registers_mapping.append(f"{reg_name}{i}")
        else:
            curr_reg_pos = self.next_reg_position
            self.next_reg_position += 1
            reg_pos[f"{reg_name}{size}"] = self.registers_position[reg_type][
                f"{reg_name}{size}"
            ] = curr_reg_pos
            self.registers_mapping.append(f"{reg_name}{size}")

        # expand columns
        if reg_type == "qreg":
            self._columns.expand_cols(size=size)
        else:
            self._columns.expand_cols(size=1)

        return reg_pos

    def add_gate(
        self, gate_name: str, qargs: list, params: dict = None, controls: list = None
    ):
        # calculate which registers the gate will be on
        reg_pos = [self.registers_mapping.index(arg) for arg in qargs]
        control_pos = (
            [self.registers_mapping.index(arg) for arg in controls] if controls else []
        )
        from_reg = min(reg_pos + control_pos)
        to_reg = max(reg_pos + control_pos) + 1
        gate_col = self._columns.find_and_add_to_empty_col(from_reg, to_reg)

        # calculate the col width
        self._columns.update_col_width(index=gate_col, new_width=self._calculate_gate_width(gate_name, params))

        # gate info
        gate_info = {
            "gate_name": gate_name,
            "params": {} if params is None else params,
            "qargs": qargs,
            "controls": [] if controls is None else controls,
            "col": gate_col,
        }
        self.gates.append(gate_info)

        return gate_info

    def add_measurement(self, qreg, creg, cbit=0):
        from_reg = self.registers_mapping.index(qreg)
        to_reg = self.registers_mapping.index(creg) + 1

        measure_col = self._columns.find_and_add_to_empty_col(from_reg, to_reg)
        self._columns.update_col_width(index=measure_col, new_width=40)

        measurement_info = {
            "col": measure_col,
            "qreg": qreg,
            "creg": creg,
            "cbit": cbit,
        }
        self.measurements.append(measurement_info)
        return measurement_info

    def add_barrier(self, qreg):
        from_reg = self.registers_mapping.index(qreg)
        to_reg = from_reg + 1

        barrier_col = self._columns.find_and_add_to_empty_col(from_reg, to_reg)
        self._columns.update_col_width(index=barrier_col, new_width=40)
        barrier_info = {
            "col": barrier_col,
            "qreg": qreg,
        }
        self.barriers.append(barrier_info)

        return barrier_info

    def add_multi_barriers(self, qreg):
        reg_pos = [self.registers_mapping.index(arg) for arg in qreg]
        from_reg = min(reg_pos)
        to_reg = max(reg_pos) + 1

        barriers_col = self._columns.find_and_add_to_empty_col(from_reg, to_reg)
        self._columns.update_col_width(index=barriers_col, new_width=40)

        for reg in qreg:
            barrier_info = {
                "col": barriers_col,
                "qreg": reg,
            }
            self.barriers.append(barrier_info)

    def add_reset(self, qreg):
        from_reg = self.registers_mapping.index(qreg)
        to_reg = from_reg + 1

        reset_col = self._columns.find_and_add_to_empty_col(from_reg, to_reg)
        self._columns.update_col_width(index=reset_col, new_width=40)
        reset_info = {
            "col": reset_col,
            "qreg": qreg,
        }
        self.resets.append(reset_info)

        return reset_info

    # Classical control gate
    def add_classical_control(self, creg, gate_name: str, qargs: list, params: dict = None):
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
        self._columns.update_col_width(index=gate_col, new_width=self._calculate_gate_width(gate_name, params))

        classical_control_info = {
            "col": gate_col,
            "creg": creg,
            "gate_info": {
                "gate_name": gate_name,
                "params": {} if params is None else params,
                "qargs": qargs,
                "control": [],
            },
        }
        self.classical_controls.append(classical_control_info)
        return classical_control_info

    def build_visualization_info(self):
        # calculate x_pos for gates and columns
        start = 30
        cols_mid_point = []

        for i, v in enumerate(self._columns.col_width):
            cols_mid_point.append(start + v / 2)
            start += v + 20

        for gate in self.gates:
            gate["x_pos"] = cols_mid_point[gate["col"]]
        for measurement in self.measurements:
            measurement["x_pos"] = cols_mid_point[measurement["col"]]
        for reset in self.resets:
            reset["x_pos"] = cols_mid_point[reset["col"]]
        for barrier in self.barriers:
            barrier["x_pos"] = cols_mid_point[barrier["col"]]
        for classical_control in self.classical_controls:
            classical_control["x_pos"] = cols_mid_point[classical_control["col"]]

        visualization_dict = {
            "registers": self.registers_position,
            "gates": self.gates,
            "measurements": self.measurements,
            "resets": self.resets,
            "barriers": self.barriers,
            "classical_controls": self.classical_controls,
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

        # add qreg and creg to painter
        for qreg in parser.ast["def"]["qreg"]:
            size = parser.ast["def"]["qreg"][qreg]["index"]
            self.add_register(qreg, size, "qreg")
        for creg in parser.ast["def"]["creg"]:
            size = parser.ast["def"]["creg"][creg]["index"]
            self.add_register(creg, size, "creg")

        # add ops to painter
        for op in parser.ast["ops"]:
            print(op)
            if op["type"] == "custom_unitary":
                qargs = [self.convert_qargs_tuple_to_str(qarg) for qarg in op["qargs"]]
                self.add_gate(str.upper(op["name"]), qargs)
            if op["type"] == "cnot":
                qargs = op["target"]["name"] + str(op["target"]["index"])
                controls = op["control"]["name"] + str(op["control"]["index"])
                self.add_gate(gate_name="CX", qargs=[qargs], controls=[controls])
            if op["type"] == "measure":
                qreg = op["qreg"]["name"] + str(op["qreg"]["index"])
                creg = op["creg"]["name"] + str(
                    parser.ast["def"]["creg"][op["creg"]["name"]]["index"]
                )
                self.add_measurement(qreg=qreg, creg=creg, cbit=op["creg"]["index"])
            if op['type'] == 'barrier':
                qreg_list = []
                for reg in op['qreg']:
                    for i in range(parser.ast['def']['qreg'][reg]['index']):
                        qreg_list.append(f"{reg}{i}")
                print(qreg_list)
            if op["type"] == "reset":
                qreg = op["name"] + str(op["index"])
                self.add_reset(qreg)
            if op["type"] == "if":
                creg = op["creg"]["name"] + str(
                    parser.ast["def"]["creg"][op["creg"]["name"]]["index"]
                )
                qargs = [
                    self.convert_qargs_tuple_to_str(op["custom_unitary"]["qargs"][0])
                ]
                self.add_classical_control(
                    creg=creg,
                    gate_name=str.upper(op["custom_unitary"]["name"]),
                    params=op["custom_unitary"]["params"],
                    qargs=qargs,
                )
        return
