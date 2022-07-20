"""
The purpose of this document is to have a single place in which openQASM functionality has to be kept up to date

TODO: consider what to do if we move onto qudits
TODO: gate definitions drawn from openQASM 3, so there's actually a global phase shift in the implementations
in openQASM 2.0 due to the ways in which things were implemented. We should fix that if we ever want to use
openQASM for anything other than visualization purposes

TODO: refactor to use: https://github.com/Qiskit/qiskit-terra/blob/main/qiskit/qasm/libs/qelib1.inc
      I think this may allow us to have all the ops we need?
"""


class OpenQASMInfo:
    """
    OpenQASM information manager

    This keeps track of the import statements useful to building a component, of
    how to formulate a gate definitions in openqasm, and of how to apply a gate between specific qubits
    """

    def __init__(self, gate_name, imports: list, definitions, usage, multi_comp, gate_symbol=None):
        """
        Create a openQASMInfo object

        :param gate_name: name of the gate (must be the name used in definitions and usage)
        :type gate_name: str
        :param gate_symbol: the symbol representing a gate in qiskit's visualization
        :type gate_symbol: str
        :param imports: a list of strings defining the imports needed to use a function
        :type imports: list (of strs)
        :param definitions: definition of one or more gate in openQASM format
                           Example: "gate x a { U(pi, 0, pi) a; }"
        :type definitions: str or list
        :param usage: A function which creates a gate string given q_reg, q_reg_type, c_reg
        :type usage: function
        :return: function returns nothing
        :rtype: None
        """
        self.gate_name = gate_name
        self.gate_symbol = gate_symbol
        self.imports = imports
        if isinstance(definitions, str):
            self.definitions = [definitions]
        else:
            self.definitions = definitions
        self.usage = usage
        self.multi_comp = multi_comp

    @property
    def import_strings(self):
        """
        Returns a list of strings with all imports needed to use a specific openqasm gate/operation

        :return: a list of import string
        :rtype: list
        """
        return [f"import {i};\n" for i in self.imports]

    @property
    def define_gate(self):
        """
        Returns a list of strings which together defines a gate

        :return: gate definitions list
        :rtype: list
        """
        if isinstance(self.definitions, str):
            return [self.definitions]
        return self.definitions

    def use_gate(self, q_registers, q_registers_type, c_registers):
        """
        Returns a string which applies a gate on the quantum registers q_registers, and the classical
        registers c_registers

        :param q_registers: quantum registers on which to apply a gate
        :type q_registers: tuple
        :param q_registers_type: type of quantum registers ('e' for emitter qubits, 'p' for photonic qubits)
        :param c_registers: classical registers on which to apply a gate
        :type c_registers: tuple
        :return: a string which applies a gate on the provided quantum registers
        :rtype: str
        """
        return self.usage(q_registers, q_registers_type, c_registers)


# -------------------------- General Helpers-------------------------------------


def openqasm_header():
    """
    OpenQASM header (must be at at the top of any scripts). We use OPENQASM 2.0
    here because OPENQASM 3.0 does not seem to be supported for qiskit visualization

    :return: the header
    :rtype: str
    """
    return "OPENQASM 2.0;"


def register_initialization_string(e_reg, p_reg, c_reg):
    """
    Given the registers of a circuit, this function returns a string initializing
    all the qreg and cregs for the openQASM circuit representation

    :param e_reg: a list of emitter qubit registers, where q_registers[i] is the size of quantum register i
    :type e_reg: list
    :param p_reg: a list of photonic qubit registers, where q_registers[i] is the size of quantum register i
    :type p_reg: list
    :param c_reg: a list of registers, where c_registers[i] is the size of classical register i
    :type c_reg: list
    :return: a string initializing the registers
    :rtype: str
    """
    register_strs = []

    for r, b in enumerate(p_reg):
        q_str = f"qreg p{r}[{b}];"
        register_strs.append(q_str)

    for r, b in enumerate(e_reg):
        q_str = f"qreg e{r}[{b}];"
        register_strs.append(q_str)

    for r, b in enumerate(c_reg):
        c_str = f"creg c{r}[{b}];"
        register_strs.append(c_str)

    return "\n".join(register_strs)


# --------------------- Gate Specific Definitions -------------------------------

"""
The following create OpenQASMInfo object for different Operation types

The Operation classes can then invoke these objects as a field 
"""


def cnot_info():
    imports = []
    definition = ""

    def usage(q_reg, q_reg_type, c_reg):
        return f"CX {q_reg_type[0]}{q_reg[0]}[0], {q_reg_type[1]}{q_reg[1]}[0];"

    return OpenQASMInfo("CX", imports, definition, usage, False)


def sigma_x_info():
    imports = []
    definition = "gate x a { U(pi, 0, pi) a; }"

    def usage(q_reg, q_reg_type, c_reg):
        return f"x {q_reg_type[0]}{q_reg[0]}[0];"

    return OpenQASMInfo("x", imports, definition, usage, False, gate_symbol="X")


def sigma_y_info():
    imports = []
    definition = "gate y a { U(pi, pi / 2, pi / 2) a; }"

    def usage(q_reg, q_reg_type, c_reg):
        return f"y {q_reg_type[0]}{q_reg[0]}[0];"

    return OpenQASMInfo("y", imports, definition, usage, False, gate_symbol="Y")


def sigma_z_info():
    imports = []
    definition = "gate z a { U(0, pi/2, pi/2) a; }"

    def usage(q_reg, q_reg_type, c_reg):
        return f"z {q_reg_type[0]}{q_reg[0]}[0];"

    return OpenQASMInfo("z", imports, definition, usage, False, gate_symbol="Z")


def hadamard_info():
    imports = []
    definition = "gate h a { U(pi/2, 0, pi) a; }"

    def usage(q_reg, q_reg_type, c_reg):
        return f"h {q_reg_type[0]}{q_reg[0]}[0];"

    return OpenQASMInfo("h", imports, definition, usage, False, gate_symbol="H")


def phase_info():
    # TODO: investigate why the code fails if we name this gate "p" instead...
    imports = []
    definition = "gate s a { U(0, pi/2, 0) a; }"

    def usage(q_reg, q_reg_type, c_reg):
        return f"s {q_reg_type[0]}{q_reg[0]}[0];"

    return OpenQASMInfo("s", imports, definition, usage, False, gate_symbol="P")


def single_qubit_wrapper_info(op_list):
    """
    This function combines the openqasm info from multiple base Operations,
    to create a composed block of multiple operations
    :param op_list: list of operation classes which will be applied by the Operation block
    :type op_list: list
    :return: OpenQASMInfo object corresponding to the composed gate
    :rtype: OpenQASMInfo
    """
    imports = []
    definitions = []
    gate_name = ""
    gate_symbol = ""
    def_usage = ""
    gate_name_dict = {"": empty_info()}

    for op_class in op_list:
        oq_info = op_class.openqasm_info()
        gate_name_dict[oq_info.gate_name] = oq_info
        if (
            oq_info.gate_name == ""
        ):  # this is a gate we don't actually need (effectively identity)
            continue

        imports = imports + oq_info.import_strings
        definitions = definitions + oq_info.define_gate
        gate_name += oq_info.gate_name
        gate_symbol += oq_info.gate_symbol
        def_usage += f"{oq_info.gate_name} a;\n"

    if gate_name in gate_name_dict:  # i.e. gate is already somehow defined
        return gate_name_dict[gate_name]

    definitions.append("gate " + gate_name + " a { \n" + def_usage + "}")

    def usage(q_reg, q_reg_type, c_reg):
        return f"{gate_name} {q_reg_type[0]}{q_reg[0]}[0];"

    return OpenQASMInfo(gate_name, imports, definitions, usage, False, gate_symbol=f"U={gate_symbol}")


def cphase_info():
    imports = []
    definition = "gate cz a, b { U(pi/2, 0, pi) b; CX a, b; U(pi/2, 0, pi) b; }"  # H on target, CX, H on target

    def usage(q_reg, q_reg_type, c_reg):
        return f"cz {q_reg_type[0]}{q_reg[0]}[0], {q_reg_type[1]}{q_reg[1]}[0];"

    return OpenQASMInfo("cz", imports, definition, usage, False, gate_symbol="CZ")


def classical_cnot_info():
    imports = []
    definition = sigma_x_info().definitions[0]

    def usage(q_reg, q_reg_type, c_reg):
        return (
            f"measure {q_reg_type[0]}{q_reg[0]}[0] -> c{c_reg[0]}[0]; \n"
            f"if (c{c_reg[0]}==1) x {q_reg_type[1]}{q_reg[1]}[0];"
        )

    return OpenQASMInfo("ccnot", imports, definition, usage, True)


def classical_cphase_info():
    imports = []
    definition = sigma_z_info().definitions[0]

    def usage(q_reg, q_reg_type, c_reg):
        return (
            f"measure {q_reg_type[0]}{q_reg[0]}[0] -> c{c_reg[0]}[0]; \n"
            f"if (c{c_reg[0]}==1) z {q_reg_type[1]}{q_reg[1]}[0];"
        )

    return OpenQASMInfo("ccphase", imports, definition, usage, True)


def measurement_cnot_and_reset():
    imports = []
    definition = sigma_x_info().definitions[0]

    def usage(q_reg, q_reg_type, c_reg):
        return (
            f"measure {q_reg_type[0]}{q_reg[0]}[0] -> c{c_reg[0]}[0]; \n"
            f"if (c{c_reg[0]}==1) x {q_reg_type[1]}{q_reg[1]}[0]; \n"
            f"barrier {q_reg_type[0]}{q_reg[0]}, {q_reg_type[1]}{q_reg[1]}; \n"
            f"reset {q_reg_type[0]}{q_reg[0]}[0];"
        )

    return OpenQASMInfo("ccnot_and_reset", imports, definition, usage, True)


def z_measurement_info():
    imports = []
    definition = ""

    def usage(q_reg, q_reg_type, c_reg):
        return f"measure {q_reg_type[0]}{q_reg[0]}[0] -> c{c_reg[0]}[0];"

    return OpenQASMInfo("measure z", imports, definition, usage, False)


def empty_info():
    return OpenQASMInfo("", [], "", lambda x, y, z: "", False)
