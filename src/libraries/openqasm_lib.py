"""
The purpose of this document is to have a single place in which openQASM functionality has to be kept up to date

TODO: maybe come up with a system that doesn't force the quantum registers to
all come before the classical ones? See if that ever becomes an issue--> if so
we could just have a quantum-specific escape character, and a classical-specific escape
character
"""

OPENQASM_ESCAPE_STR = '%%%'


class OpenQASMInfo:
    """
    OpenQASM information manager
    """
    def __init__(self, imports:list, definition:str, usage: str):
        self.imports = list(imports)
        self.definition = definition
        self.usage_func = self.apply_gate_string(usage)

    @staticmethod
    def apply_gate_string(usage):
        def apply_gate(q_registers, c_registers):
            gate_str = usage
            for q in q_registers:
                reg_str = f'q{q[0]}[{q[1]}]'
                gate_str = gate_str.replace(OPENQASM_ESCAPE_STR, reg_str, 1)

            for c in c_registers:
                reg_str = f'c{c[0]}[{c[1]}]'
                gate_str = gate_str.replace(OPENQASM_ESCAPE_STR, reg_str, 1)

            assert OPENQASM_ESCAPE_STR not in gate_str  # check that all escapes have been replaced
            return gate_str

        return apply_gate

    @property
    def import_string(self):
        return "\n".join(self.imports)

    @property
    def define_gate(self):
        return self.definition


def cnot_info():
    imports = ["stdgates.inc"]
    definition = ""
    usage = f"cx {OPENQASM_ESCAPE_STR}, {OPENQASM_ESCAPE_STR};"

    return OpenQASMInfo(imports, definition, usage)


def sigma_x_info():
    imports = ["stdgates.inc"]
    definition = ""
    usage = f"x {OPENQASM_ESCAPE_STR};"

    return OpenQASMInfo(imports, definition, usage)


def sigma_y_info():
    imports = ["stdgates.inc"]
    definition = ""
    usage = f"y {OPENQASM_ESCAPE_STR};"

    return OpenQASMInfo(imports, definition, usage)


def sigma_z_info():
    imports = ["stdgates.inc"]
    definition = ""
    usage = f"z {OPENQASM_ESCAPE_STR};"

    return OpenQASMInfo(imports, definition, usage)


def sigma_z_info():
    imports = ["stdgates.inc"]
    definition = ""
    usage = f"h {OPENQASM_ESCAPE_STR};"

    return OpenQASMInfo(imports, definition, usage)


def empty_info():
    return OpenQASMInfo([], "", "")
