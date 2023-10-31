import json
import requests
from graphiq.utils.openqasm_parser import OpenQASMParser

standard_gate_width_mapping = {
    "CX": 40,
    "CZ": 40,
}


class Columns:
    """
    This class is used to manage the positions of the operations in the Painter class below. When the Painter
    class adds operations, it will check for empty column and add operation there. This class is also used for manage the
    width of each column.

    The structure of the Columns class is a 2d array. First, it includes an array of columns since we will have multiple
    columns in the figure. The size of each column is equal to the number of registers (qreg and creg).
    """

    def __init__(self, col_num: int, size: int = 1):
        """
        The Columns class constructor. Initialize a number of columns according to the col_num parameter.

        Each column has the same size specified by the size parameter.

        Also create a col_width variable to maintain the width of each column

        :param col_num: The number of columns
        :type col_num: int
        :param size: The size for each column
        :type size: int
        """
        self.size = size
        self.columns = []

        for i in range(col_num):
            self.columns.append([0] * size)

        self.col_width = [0] * col_num

    def add_new_column(self):
        """
        Add a column to the columns variable with the same size as other columns

        :return: nothing
        :rtype: None
        """
        new_column = [0] * self.size
        self.columns.append(new_column)
        self.col_width.append(0)

    def update_col_width(self, index, new_width):
        """
        Update the column width of a column. Mostly used when encountering a gate that has bigger width to draw than
        other gates.

        :param index: index of the column that need to update the width
        :type index: int
        :param new_width: new width of the column
        :type: new_width: int
        :return: nothing
        :rtype: None
        """

        self.col_width[index] = max(self.col_width[index], new_width)

    def expand_cols(self, size: int = 1):
        """
        Expand all columns to a new size. It is used when the Painter adds a new register.

        :param size: The size that all columns will be expanded
        :type size: int
        :return: nothing
        :rtype: None
        """

        if not self.columns:
            self.columns.append([0] * size)
            self.col_width.append(0)
        else:
            for col in self.columns:
                col.extend([0] * size)
        self.size += size

    def set_all_col_element(self, index: int, value: int = -1):
        """
        Set all the value in every column from the first column to the index column.

        This function is used when adding barriers,
        since after adding barriers, new operations will be added after the barriers.

        :param index: The index of the column
        :type index: int
        :param value: The value that all column value will be set to if current value is not 0
        :type value: int
        :return: nothing
        :rtype: None
        """

        if not isinstance(index, int):
            raise ValueError("Index parameter must be an integer")
        for i in reversed(range(len(self.columns))):
            if not self.columns[i][index]:
                self.columns[i][index] = value
        return

    def find_and_add_to_empty_col(self, from_index: int, to_index: int, value: int = 1):
        """
        Find the column that the operation can be added to. If no column found, add a new column at the end and
        set value to that column.

        :param from_index: The index the operation start from
        :type from_index: int
        :param to_index: The index the operation end at
        :type to_index: int
        :param value: The value that to be set at the correct column
        :return: The column that the operation is added to
        :rtype: int
        """
        col = 0

        for i, v in enumerate(self.columns):
            gate_pos = v[from_index:to_index]

            if i == len(self.columns) - 1 and not all(pos == 0 for pos in gate_pos):
                self.add_new_column()
                self.columns[i + 1][from_index:to_index] = [value] * len(gate_pos)
                col = i + 1
                break
            if all(pos == 0 for pos in gate_pos):
                self.columns[i][from_index:to_index] = [value] * len(gate_pos)
                col = i
                break

        to_index = to_index - 1
        if to_index != from_index:
            for i in reversed(range(col)):
                if not self.columns[i][from_index]:
                    self.columns[i][from_index] = -1
                if not self.columns[i][to_index]:
                    self.columns[i][to_index] = -1
        return col


class Painter:
    """
    The main goal of the Painter class is to create the visualization information that will be sent to the web app
    """

    def __init__(self, gate_mapping=None):
        """
        Painter class constructor.

        It can take no argument or gate mapping.

        :param gate_mapping: gate mapping to specify the gate width mapping
        :type gate_mapping: dict
        """

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
        """
        A function that will generate the register label from register name and size

        :param reg_name: register name
        :type reg_name: str
        :param size: the size of the register
        :type size: int
        :return: a string that represent the label mapping of the register
        rtype: str
        """
        return str(reg_name) + "[" + str(size) + "]"

    def _calculate_gate_width(self, gate_name, params):
        """
        A helper function that is used to calculate the gate width,
        since a gate with a long name or with params has different width when it is drawn.

        :param gate_name: gate name
        :type gate_name: str
        :param params: params of the gate
        :type params: dict
        :return: the calculated width of a gate
        :rtype: int
        """
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
        """
        A function that add new register to the Painter. This function will create the correct register label to draw in
        the html figure, with a position mapping to that label.

        :param reg_name: register name
        :type reg_type: str
        :param size: size of the register
        :type size: int
        :param reg_type: type of the register (can only be "qreg" or "creg")
        :type reg_type: str
        :return: nothing
        :rtype: None
        """
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

    def add_gate(
        self, gate_name: str, qargs: list, params: dict = None, controls: list = None
    ):
        """
        Add a gate to in the Painter class. This function constructs the correct position to draw the gate in a
        html figure.

        :param gate_name: gate name
        :type gate_name: str
        :param qargs: qargs of thge gate
        :type qargs: list
        :param params: params of the gate
        :type params: dict
        :param controls: gate controls for gate like CX, it will specify where the control at
        :return: visualization info of the gate
        rtype: dict
        """
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
        """
        Add measurement to the Painter class. The function constructs the correct position for the measurement, the
        visualization info will include also the creg position to draw correctly in a html figure.

        :param qreg: qreg of the measurement
        :type qreg: str
        :param creg: creg of the measurement
        :type creg: str
        :param cbit: the cbit measure to in the creg
        :type cbit: int
        :return: measurement visualization info
        :rtype: dict
        """

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
        """
        Add barriers to the Painter class. The function constructs correct positions for the barriers.

        :param qreg: list of qreg
        :type qreg: list
        :return: nothing
        :rtype: None
        """
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
        """
        Add reset to the Painter class. The function constructs correct position for the reset to draw on the html
        figure.

        :param qreg: qreg
        :type qreg: str
        :return: reset visualization info
        :rtype: dict
        """

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
        """
        Add classical control to the Painter class. The function constructs visualization info for classical control
        operation.

        :param creg: creg
        :type creg: str
        :param gate_name: gate name
        :type gate_name: str
        :param qargs: qargs of the gate
        :type qargs: list
        :param params: params of the gate
        :type params: dict
        :return: visualization info of the classical control
        :rtype: dict
        """

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
        """
        The function that constructs a visualization dict of the circuit including operations info.

        The function calculates the x coordinate of each operation.

        :return: visualization of the circuit
        :rtype: dict
        """
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
        """
        Send the visualization info of the circuit to the web app.

        The default connection will be localhost at port 5000 and the parameter is '/circuit_data'

        In the future if we have a web server we can consider, sending data to the web server, and the visualization
        will be display on the website.

        :return: request response class indicate the status of the request
        :rtype: requests.Response
        """
        visualization_info = self.build_visualization_info()
        url = "http://127.0.0.1:5000/circuit_data"

        data = json.dumps(visualization_info, indent=3)
        result = requests.post(url, data=data)

        return result

    def load_openqasm_str(self, openqasm_str):
        """
        Translate the OpenQASM 2.0 script to visualization info.

        :param openqasm_str: OpenQASM 2.0 script string
        :type openqasm_str: str
        :return: nothing
        :rtype: None
        """
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
