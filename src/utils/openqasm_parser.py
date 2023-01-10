from qiskit.qasm import Qasm


# TODO: Add docstring
class OpenQASMParser:
    """
    This parser just use the Qiskit parser to parse the openqasm string, but the information is more readable.
    """

    def __init__(self, openqasm_str: str):
        self.openqasm = openqasm_str

        self.ast = {
            "def": {
                "qreg": {},
                "creg": {},
                "gate": {},
            },
            "ops": [],
        }

        # mapping parse functions
        self._parse_mapping = {
            'qreg': self._parse_qreg,
            'creg': self._parse_creg,
            'gate': self._parse_gate,
            'barrier': self._parse_barrier,
            'reset': self._parse_reset,
            'measure': self._parse_measure,
            'cnot': self._parse_cnot,
            'custom_unitary': self._parse_custom_unitary,
            'if': self._parse_if,
        }

    def use_parse(self, key):
        return self._parse_mapping[key]

    def get_register_size(self, reg_name, reg_type):
        return self.ast['def'][reg_type][reg_name]['size']

    def parse(self):
        parser = Qasm(data=self.openqasm).parse()
        def_arr = ['qreg', 'creg', 'gate']

        for node in parser.children:
            # add to def
            if node.type not in self._parse_mapping:
                continue
            parse_function = self.use_parse(key=node.type)
            info = parse_function(node)

            if node.type in def_arr:
                self.ast["def"][node.type][node.name] = info
            else:
                self.ast["ops"].append(info)
            yield info

    # TODO: Maybe add parse body, because in the gate body there is a list of statements
    #  The parser originally is created for visualization purpose, but right now parsing the body of a gate
    #  doesn't help much. Therefore, in order to save computer resource, I just ignore parsing the body.
    def _parse_gate(self, gate_node):
        gate_info = {
            "type": gate_node.type,
            "name": gate_node.name,
            "params": [],
            "qargs": [],
        }

        # Get params and qargs
        # Go to 1st children node, nodes that not is_bit is param, node that is_bit is qargs
        for node in gate_node.children[1].children:
            if node.is_bit:
                gate_info["qargs"].append(node.name)
            else:
                gate_info["params"].append(node.name)

        # some qarg is in the 2nd children node, don't know why
        for node in gate_node.children[2].children:
            if node.type == "id":
                gate_info["qargs"].append(node.name)
        return gate_info

    def _parse_creg(self, creg_node):
        creg_info = {
            "type": creg_node.type,
            "size": creg_node.index,
            "name": creg_node.name,
        }
        return creg_info

    def _parse_qreg(self, qreg_node):
        qreg_info = {
            "type": qreg_node.type,
            "size": qreg_node.index,
            "name": qreg_node.name,
        }
        return qreg_info

    def _parse_barrier(self, barrier_node):
        barrier_info = {"type": barrier_node.type, "qreg": []}
        for node in barrier_node.children:
            for children_node in node.children:
                barrier_info["qreg"].append(children_node.name)
        return barrier_info

    def _parse_reset(self, reset_node):
        reset_info = {
            "type": reset_node.type,
        }
        for node in reset_node.children:
            reset_info["name"] = node.name
            if hasattr(node, "index"):
                reset_info["index"] = node.index
        return reset_info

    def _parse_expression(self, expression_node):
        expression_info = {}
        if expression_node.type == "binop":
            for binop_node in expression_node.children:
                if binop_node.type == "operator":
                    expression_info["operator"] = binop_node.value
                if binop_node.type == "real":
                    expression_info["real"] = binop_node.value
                if binop_node.type == "int":
                    expression_info["int"] = binop_node.value
        else:
            expression_info[expression_node.type] = expression_node.value
        return expression_info

    def _parse_custom_unitary(self, custom_unitary_node):
        custom_unitary_info = {
            "type": custom_unitary_node.type,
            "name": custom_unitary_node.name,
            "params": {},
            "qargs": [],
        }
        params_mapping = self.ast["def"]["gate"][custom_unitary_node.name]["params"]

        for node in custom_unitary_node.children:
            # get params
            if node.type == "expression_list":
                for i, exp_node in enumerate(node.children):
                    custom_unitary_info["params"][
                        params_mapping[i]
                    ] = self._parse_expression(exp_node)
            if node.type == "primary_list":
                for p_node in node.children:
                    qarg = (p_node.name, p_node.index)
                    custom_unitary_info["qargs"].append(qarg)
        return custom_unitary_info

    def _parse_measure(self, measure_node):
        measure_info = {
            "type": measure_node.type,
        }

        qreg_node = measure_node.children[0]
        creg_node = measure_node.children[1]

        measure_info["qreg"] = {
            "name": qreg_node.name,
        }
        if hasattr(qreg_node, "index"):
            measure_info["qreg"]["index"] = qreg_node.index

        measure_info["creg"] = {
            "name": creg_node.name,
        }
        if hasattr(creg_node, "index"):
            measure_info["creg"]["index"] = qreg_node.index

        return measure_info

    def _parse_cnot(self, cnot_node):
        cnot_info = {
            "type": cnot_node.type,
            "control": {
                "name": cnot_node.children[0].name,
                "index": cnot_node.children[0].index,
            },
            "target": {
                "name": cnot_node.children[1].name,
                "index": cnot_node.children[1].index,
            },
        }
        return cnot_info

    def _parse_if(self, if_node):
        if_info = {
            "type": if_node.type,
            "creg": {
                "name": if_node.children[0].name,
                "value": if_node.children[1].value,
            },
            "custom_unitary": self._parse_custom_unitary(if_node.children[2]),
        }
        return if_info

