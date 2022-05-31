"""
The purpose of this document is to have a single place in which openQASM functionality has to be kept up to date

TODO: maybe come up with a system that doesn't force the quantum registers to
all come before the classical ones? See if that ever becomes an issue--> if so
we could just have a quantum-specific escape character, and a classical-specific escape
character
TODO: consider what to do if we move onto qudits
TODO: gate definitions drawn from openQASM 3, so there's actually a global phase shift in the implementations
in openQASM 2.0 due to the ways in which things were implemented. We should fix that if we ever want to use
openQASM for anything other than visualization purposes
"""

OPENQASM_ESCAPE_STR = '%%%'


class OpenQASMInfo:
    """
    OpenQASM information manager

    This keeps track of the import statements useful to building a component, of
    how to formulate a gate definition in openqasm, and of how to apply a gate between specific qubits
    """
    def __init__(self, imports: list, definition: str, usage: str):
        """
        Create a openQASMInfo object

        :param imports: a list of strings defining the imports needed to use a function
        :type imports: list (of strs)
        :param definition: definition of a gate in openQASM format
                           Example: "gate x a { U(pi, 0, pi) a; }"
        :type definition: str
        :param usage: A string explaining how to create a given gate
                      Example: f"x {OPENQASM_ESCAPE_STR}; where OPENQASM_ESCAPE_STR
                      is replaced by the appropriate registers
        :type usage: str
        :return: function returns nothing
        :rtype: None
        """
        self.imports = list(imports)
        self.definition = definition
        self.usage = usage

    @property
    def import_string(self):
        """
        Returns a string with all imports needed to use a specific openqasm gate/operation

        :return: an import string
        :rtype: str
        """
        return "\n".join([f'import {i};' for i in self.imports])

    @property
    def define_gate(self):
        """
        Returns a string which defines a gate

        :return: gate definition string
        :rtype: str
        """
        return self.definition

    def use_gate(self, q_registers, c_registers):
        """
        Returns a string which applies a gate on the quantum registers q_registers, and the classical
        registers c_registers

        :param q_registers: quantum registers on which to apply a gate
        :type q_registers: tuple
        :param c_registers: classical registers on which to apply a gate
        :type c_registers: tuple
        :return: a string which applies a gate on the provided quantum registers
        :rtype: str
        """
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
    """
    OpenQASM header (must be at at the top of any scripts). We use OPENQASM 2.0
    here because OPENQASM 3.0 does not seem to be supported for qiskit visualization

    :return: the header
    :rtype: str
    """
    return 'OPENQASM 2.0;'


def register_initialization_string(q_registers, c_registers):
    """
    Given the registers of a circuit, this function returns a string initializing
    all the qreg and cregs for the openQASM circuit representation

    :param q_registers: a list of registers, where q_registers[i] is the size of quantum register i
    :type q_registers: list
    :param c_registers: a list of registers, where c_registers[i] is the size of classical register i
    :type c_registers: list
    :return: a string initializing the registers
    :rtype: str
    """
    register_strs = []
    for r, b in enumerate(q_registers):
        q_str = f'qreg q{r}[{b}];'
        register_strs.append(q_str)

    for r, b in enumerate(c_registers):
        c_str = f'creg c{r} [{b}];'
        register_strs.append(c_str)

    return '\n'.join(register_strs)


# --------------------- Gate Specific Definitions -------------------------------

"""
The following create OpenQASMInfo object for different Operation types

The Operation classes can then invoke these objects as a field 
"""


def cnot_info():
    imports = []
    definition = ""
    usage = f"CX {OPENQASM_ESCAPE_STR}, {OPENQASM_ESCAPE_STR};"

    return OpenQASMInfo(imports, definition, usage)


def sigma_x_info():
    imports = []
    definition = "gate x a { U(pi, 0, pi) a; }"
    usage = f"x {OPENQASM_ESCAPE_STR};"

    return OpenQASMInfo(imports, definition, usage)


def sigma_y_info():
    imports = []
    definition = "gate y a { U(pi, pi / 2, pi / 2) a; }"
    usage = f"y {OPENQASM_ESCAPE_STR};"

    return OpenQASMInfo(imports, definition, usage)


def sigma_z_info():
    imports = []
    definition = "gate z a { U(0, pi/2, pi/2) a; }"
    usage = f"z {OPENQASM_ESCAPE_STR};"

    return OpenQASMInfo(imports, definition, usage)


def hadamard_info():
    imports = []
    definition = "gate h a { U(pi/2, 0, pi) a; }"
    usage = f"h {OPENQASM_ESCAPE_STR};"

    return OpenQASMInfo(imports, definition, usage)


def cphase_info():
    imports = []
    definition = "gate cz a, b { U(pi/2, 0, pi) b; CX a, b; U(pi/2, 0, pi) b; }"  # H on target, CX, H on target
    usage = f"cz {OPENQASM_ESCAPE_STR}, {OPENQASM_ESCAPE_STR};"

    return OpenQASMInfo(imports, definition, usage)


def classical_cnot_info():
    imports = []
    definition = "gate cCX a, b, c { }"
    usage = f"cCX {OPENQASM_ESCAPE_STR}, {OPENQASM_ESCAPE_STR} -> {OPENQASM_ESCAPE_STR};"
    raise NotImplementedError("Classical-quantum gates not particularly compatible with openQASM 2.0, not implemented yet")


def classical_cphase_info():
    imports = []
    definition = "gate ccphase a, b, c { }"
    usage = f"ccphase {OPENQASM_ESCAPE_STR}, {OPENQASM_ESCAPE_STR} -> {OPENQASM_ESCAPE_STR};"
    raise NotImplementedError("Classical-quantum gates not particularly compatible with openQASM 2.0, not implemented yet")


def z_measurement_info():
    imports = []
    definition = ""
    usage = f"measure {OPENQASM_ESCAPE_STR} -> {OPENQASM_ESCAPE_STR};"

    return OpenQASMInfo(imports, definition, usage)


def empty_info():
    return OpenQASMInfo([], "", "")
