class Painter:
    def __init__(self):
        self.next_register_height = 50
        self.info = {
            "label_width": 70,
            "label_height": 50,
            "detail_width": 50,
            "detail_height": 50,
            "last_col": 0,
        }

        self.registers_height = {}
        self.registers_col = {}
        self.registers_order = []
        self.gates = []
        self.col_width = {}

    def add_new_col(self):
        last_col = self.info["last_col"]
        self.registers_col[last_col + 1] = [0] * len(self.registers_height)
        self.info["last_col"] += 1

    def add_register(self, register, reg_type):
        curr_reg = self.next_register_height
        self.next_register_height += 50
        self.info["label_height"] += 50
        self.info["detail_height"] += 50

        self.registers_height[f"{reg_type}{register}"] = curr_reg
        self.registers_order.append(f"{reg_type}{register}")

        if self.registers_col:
            for col in self.registers_col:
                self.registers_col[col].append(0)
        else:
            self.registers_col[1] = [0]
            self.info["last_col"] += 1

        return curr_reg

    def add_gate(self, gate_name: str, qargs: list, params: list=None, control: list=None):
        reg_pos = [self.registers_order.index(arg) for arg in qargs]
        from_reg = min(reg_pos)
        to_reg = max(reg_pos)+1
        gate_col = 0

        # decide which col the gate will be in
        for col in self.registers_col:
            gate_pos = self.registers_col[col][from_reg:to_reg]

            if col == self.info["last_col"] and not all(pos == 0 for pos in gate_pos):
                self.add_new_col()
                self.registers_col[col+1][from_reg:to_reg] = [1] * len(gate_pos)
                gate_col = col + 1
                break
            if all(pos == 0 for pos in gate_pos):
                self.registers_col[col][from_reg:to_reg] = [1] * len(gate_pos)
                gate_col = col
                break

        # calculate the col width
        params_str = ""
        if params is not None:
            params_str += "("
            params_str += ', '.join(map(str, params))
            params_str += ")"
        width = max(40, 15*2 + len(gate_name)*10, 15*2 + len(params_str)*5)

        if gate_col not in self.col_width:
            self.col_width[gate_col] = width
        else:
            self.col_width[gate_col] = max(width, self.col_width[gate_col])

        # gate info
        gate_info = {
            "gate_name": gate_name,
            "params": params_str,
            "qargs": qargs,
            "controls": [] if control is None else control,
            "col": gate_col,
            "y_pos": [self.registers_height[arg] for arg in qargs],
        }
        self.gates.append(gate_info)

        return gate_info

    def add_one_qubit_gate(self, reg_type, register, gate_name, params=None,):
        reg_pos = self.registers_order.index(f"{reg_type}{register}")
        gate_col = 0

        for col in self.registers_col:
            if col == self.info["last_col"] and self.registers_col[col][reg_pos] == 1:
                self.add_new_col()
                self.registers_col[col + 1][reg_pos] = 1
                gate_col = col + 1
                break
            if self.registers_col[col][reg_pos] == 0:
                self.registers_col[col][reg_pos] = 1
                gate_col = col
                break

        gate_info = {
            "gate_name": gate_name,
            "params": params,
            "qargs": [f"{reg_type}{register}"],
            "x_pos": gate_col * 50,
            "y_pos": self.registers_height[f"{reg_type}{register}"]
        }
        self.gates.append(gate_info)

        return gate_info

    def add_two_qubit_gate(self, control_type, control, target_type, target):
        control_reg = self.registers_order.index(f"{control_type}{control}")
        target_reg = self.registers_order.index(f"{target_type}{target}")
        from_reg = min(control_reg, target_reg)
        to_reg = max(control_reg, target_reg)

        for col in self.registers_col:
            gate_pos = self.registers_col[col][from_reg:to_reg+1]

            if col == self.info["last_col"] and not all(pos == 0 for pos in gate_pos):
                self.add_new_col()
                self.registers_col[col+1][from_reg:to_reg + 1] = [1] * len(gate_pos)
                break
            if all(pos == 0 for pos in gate_pos):
                self.registers_col[col][from_reg:to_reg+1] = [1] * len(gate_pos)
                break

    def build_visualization_info(self):
        # calculate x_pos for gates and columns
        start = 30
        cols_mid_point = [start]
        for col in self.col_width:
            cols_mid_point.append(start + self.col_width[col]/2)
            start += self.col_width[col] + 10
        for gate in self.gates:
            gate["x_pos"] = cols_mid_point[gate["col"]]

        visualization_dict = {
            "register_height": self.registers_height,
            "gates": self.gates,
        }

        return visualization_dict


