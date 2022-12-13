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
                "qreg": [],
                "creg": [],
                "gate": [],
            },
            "ops": []
        }

    def parse(self):
        parser = Qasm(data=self.openqasm).parse()

        for node in parser.children:
            # print(node.type)
            # add to def
            if node.type == 'qreg':
                self.ast["def"]["qreg"].append(self.parse_qreg_info(node))
            if node.type == 'creg':
                self.ast["def"]["creg"].append(self.parse_creg_info(node))

            # add to ops
            if node.type == 'barrier':
                self.ast["ops"].append(self.parse_barrier(node))
            if node.type == 'reset':
                self.ast["ops"].append(self.parse_reset(node))
            if node.type == 'measure':
                self.ast["ops"].append(self.parse_measure(node))
            if node.type == 'cnot':
                self.ast["ops"].append(self.parse_cnot(node))
            if node.type == 'custom_unitary':
                self.ast["ops"].append(self.parse_custom_unitary(node))
            if node.type == 'if':
                self.ast["ops"].append(self.parse_if(node))

    @staticmethod
    def parse_qreg_info(qreg_node):
        qreg_info = {
            'name': qreg_node.name,
            'index': qreg_node.index,
        }

        return qreg_info

    @staticmethod
    def parse_gate(gate_node):
        gate_info = {
        }
        return gate_info

    @staticmethod
    def parse_creg_info(creg_node):
        qreg_info = {
            'name': creg_node.name,
            'index': creg_node.index,
        }
        return qreg_info

    @staticmethod
    def parse_barrier(barrier_node):
        barrier_info = {
            'type': barrier_node.type,
            "qreg": []
        }
        for node in barrier_node.children:
            for children_node in node.children:
                barrier_info['qreg'].append(children_node.name)
        return barrier_info

    @staticmethod
    def parse_reset(reset_node):
        reset_info = {
            'type': reset_node.type,
        }
        for node in reset_node.children:
            reset_info['name'] = node.name
            if hasattr(node, 'index'):
                reset_info["index"] = node.index
        return reset_info

    @staticmethod
    def parse_custom_unitary(custom_unitary_node):
        def parse_expression(expression_node):
            expression_info = {}
            if expression_node.type == 'binop':
                for binop_node in expression_node.children:
                    if binop_node.type == 'operator':
                        expression_info['operator'] = binop_node.value
                    if binop_node.type == 'real':
                        expression_info['real'] = binop_node.value
                    if binop_node.type == 'int':
                        expression_info['int'] = binop_node.value
            else:
                expression_info[expression_node.type] = expression_node.value
            return expression_info

        custom_unitary_info = {
            'type': custom_unitary_node.type,
            'name': custom_unitary_node.name,
            'params': [],
            'qargs': [],
        }

        for node in custom_unitary_node.children:
            if node.type == 'expression_list':
                for exp_node in node.children:
                    custom_unitary_info['params'].append(parse_expression(exp_node))
            if node.type == 'primary_list':
                for p_node in node.children:
                    qarg = (p_node.name, p_node.index)
                    custom_unitary_info['qargs'].append(qarg)

        return custom_unitary_info

    @staticmethod
    def parse_measure(measure_node):
        measure_info = {
            'type': measure_node.type,
        }

        qreg_node = measure_node.children[0]
        creg_node = measure_node.children[1]

        measure_info['qreg'] = {
            'name': qreg_node.name,
        }
        if hasattr(qreg_node, 'index'):
            measure_info['qreg']['index'] = qreg_node.index

        measure_info['creg'] = {
            'name': creg_node.name,
        }
        if hasattr(creg_node, 'index'):
            measure_info['creg']['index'] = qreg_node.index

        return measure_info

    @staticmethod
    def parse_cnot(cnot_node):
        cnot_info = {
            'type': cnot_node.type,
            'control': {
                'name': cnot_node.children[0].name,
                'index': cnot_node.children[0].index,
            },
            'target': {
                'name': cnot_node.children[1].name,
                'index': cnot_node.children[1].index,
            },
        }
        return cnot_info

    @staticmethod
    def parse_if(if_node):
        if_info = {
            type: if_node.type,
        }

        print(vars(if_node))

        return if_info

