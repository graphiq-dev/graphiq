"""
The purpose of this document is to have a single place in which openQASM functionality has to be kept up to date

TODO: maybe come up with a system that doesn't force the quantum registers to
all come before the classical ones? See if that ever becomes an issue--> if so
we could just have a quantum-specific escape character, and a classical-specific escape
character

TODO: consider what to do if we move onto qudits?
"""

OPENQASM_ESCAPE_STR = '%%%'
STD_GATE_LIB = '"stdgates.inc"'

class OpenQASMInfo:
    """
    OpenQASM information manager
    """
    def __init__(self, imports:list, definition:str, usage: str):
        self.imports = list(imports)
        self.definition = definition
        self.usage = usage

    @property
    def import_string(self):
        return "\n".join([f'import {i};' for i in self.imports])

    @property
    def define_gate(self):
        return self.definition

    def use_gate(self, q_registers, c_registers):
        gate_str = self.usage
        for q in q_registers:
            reg_str = f'q{q[0]}[{q[1]}]'
            gate_str = gate_str.replace(OPENQASM_ESCAPE_STR, reg_str, 1)

        for c in c_registers:
            reg_str = f'c{c[0]}[{c[1]}]'
            gate_str = gate_str.replace(OPENQASM_ESCAPE_STR, reg_str, 1)

        assert OPENQASM_ESCAPE_STR not in gate_str  # check that all escapes have been replaced
        return gate_str
# -------------------------- General Helpers-------------------------------------


def openqasm_header():
    return 'OPENQASM 3.0;'


def register_initialization_string(q_registers, c_registers):
    register_strs = []
    for r, b in enumerate(q_registers):
        q_str = f'qubit[{b}] q{r};'
        register_strs.append(q_str)

    for r, b in enumerate(c_registers):
        c_str = f'bit[{b}] c{r};'
        register_strs.append(c_str)

    return '\n'.join(register_strs)


# --------------------- Gate Specific Definitions -------------------------------

def cnot_info():
    imports = [STD_GATE_LIB]
    definition = ""
    usage = f"cx {OPENQASM_ESCAPE_STR}, {OPENQASM_ESCAPE_STR};"

    return OpenQASMInfo(imports, definition, usage)


def sigma_x_info():
    imports = [STD_GATE_LIB]
    definition = ""
    usage = f"x {OPENQASM_ESCAPE_STR};"

    return OpenQASMInfo(imports, definition, usage)


def sigma_y_info():
    imports = [STD_GATE_LIB]
    definition = ""
    usage = f"y {OPENQASM_ESCAPE_STR};"

    return OpenQASMInfo(imports, definition, usage)


def sigma_z_info():
    imports = [STD_GATE_LIB]
    definition = ""
    usage = f"z {OPENQASM_ESCAPE_STR};"

    return OpenQASMInfo(imports, definition, usage)


def hadamard_info():
    imports = [STD_GATE_LIB]
    definition = ""
    usage = f"h {OPENQASM_ESCAPE_STR};"

    return OpenQASMInfo(imports, definition, usage)


def empty_info():
    return OpenQASMInfo([], "", "")
