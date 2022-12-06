standard_gate_mapping = {
    "CX": {
        "width": 40,
    },
    "CZ": {
        "width": 40,
    },
    "SWAP": {
        "width": 40,
    }
}


class Painter:
    def __init__(self, gate_mapping=None):
        self.next_reg_position = 50
        self.last_col = 0

        self.registers_position = {
            "qreg": {},
            "creg": {},
        }
        self.columns = []
        self.registers_mapping = []
        self.gates = []
        self.col_width = []

        if not gate_mapping:
            self.gate_mapping = standard_gate_mapping
        else:
            self.gate_mapping = gate_mapping

    def add_new_col(self):
        self.columns.append([0] * (len(self.registers_position["qreg"]) + len(self.registers_position["creg"])))
        self.col_width.append(0)
        self.last_col += 1

    def add_register(self, reg_name, size, reg_type="qreg"):
        if reg_type != "qreg" and reg_type != "creg":
            raise ValueError("Register type must be creg or qreg")

        # map register position to the correct coordinates
        if reg_type == "qreg":
            for i in range(size):
                curr_reg_pos = self.next_reg_position
                self.next_reg_position += 50
                self.registers_position[reg_type][f"{reg_name}{i}"] = curr_reg_pos
                self.registers_mapping.append(f"{reg_name}{i}")
        else:
            curr_reg_pos = self.next_reg_position
            self.next_reg_position += 50
            self.registers_position[reg_type][f"{reg_name}{size}"] = curr_reg_pos
            self.registers_mapping.append(f"{reg_name}{size}")

        # update columns
        if not self.columns:
            self.columns.append([0] * size) if reg_type == "qreg" else self.columns.append(0)
            self.col_width.append(0)
        else:
            for col in self.columns:
                col.append(0 * size) if reg_type == "qreg" else col.append(0)

        return self.registers_position

    def add_gate(self, gate_name: str, qargs: list, params: dict=None, control: list=None):
        # calculate which registers the gate will be on
        reg_pos = [self.registers_mapping.index(arg) for arg in qargs]
        from_reg = min(reg_pos)
        to_reg = max(reg_pos)+1
        gate_col = 0

        # decide which col the gate will be on
        for i, v in enumerate(self.columns):
            gate_pos = v[from_reg:to_reg]

            if i == self.last_col and not all(pos == 0 for pos in gate_pos):
                self.add_new_col()
                self.columns[i+1][from_reg:to_reg] = [1] * len(gate_pos)
                gate_col = i + 1
                break
            if all(pos == 0 for pos in gate_pos):
                self.columns[i][from_reg:to_reg] = [1] * len(gate_pos)
                gate_col = i
                break

        # calculate the col width
        params_str = ""
        if params:
            params_str += "("
            for i in params:
                params_str += f"{params[i]}, "
            params_str = params_str[:-2]
            params_str += ")"
        width = self.gate_mapping[gate_name]["width"] if gate_name.upper() in self.gate_mapping \
            else max(40, 15*2 + len(gate_name)*10, 15*2 + len(params_str)*5)
        self.col_width[gate_col] = max(width, self.col_width[gate_col])

        # gate info
        gate_info = {
            "gate_name": gate_name,
            "params": {} if params is None else params,
            "qargs": qargs,
            "controls": [] if control is None else control,
            "col": gate_col,
        }
        self.gates.append(gate_info)

        return gate_info

    def build_visualization_info(self):
        # calculate x_pos for gates and columns
        start = 30
        cols_mid_point = []

        for i, v in enumerate(self.col_width):
            cols_mid_point.append(start + v/2)
            start += v + 10
        for gate in self.gates:
            gate["x_pos"] = cols_mid_point[gate["col"]]

        visualization_dict = {
            "registers": self.registers_position,
            "gates": self.gates,
        }

        return visualization_dict

