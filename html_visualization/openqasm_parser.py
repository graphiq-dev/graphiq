import re


# TODO: consider using a parser generator library for easier management?
# TODO: Add handle statement for if, measurement, barrier, and reset
class OpenQASMParser:
    def __init__(self, openqasm_str: str):
        self.openqasm = openqasm_str

        self.ast = {
            "def": {
                "qreg": {},
                "creg": {},
                "gate": {},
                "opaque": {},
            },
            "ops": {

            }
        }

    # TODO: Use RegEx to parse str
    def parse(self):
        statements = self.openqasm.split(';')

        for i, v in enumerate(statements):
            statements[i] = v.replace('\n', "")
            words = v.split()

            if words and words[0] == "qreg":
                self.parse_qreg(words[1])
            if words and words[0] == "creg":
                self.parse_creg(words[1])

        print(statements)

    def parse_qreg(self, statement):
        q_name = re.search(pattern="^[a-zA-Z0-9\_]+", string=statement).group()
        q_size = re.search(pattern="\[[0-9]+\]", string=statement).group()
        q_size = re.search(pattern="[0-9]+", string=q_size).group()

        self.ast["def"]["qreg"][f"{q_name}"] = q_size

    def parse_creg(self, statement):
        c_name = re.search(pattern="^[a-zA-Z0-9\_]+", string=statement).group()
        c_size = re.search(pattern="\[[0-9]+\]", string=statement).group()
        c_size = re.search(pattern="[0-9]+", string=c_size).group()

        self.ast["def"]["creg"][f"{c_name}"] = c_size

    def get_gate_definition(self):
        return

    def get_qreg(self):
        return

    def get_creg(self):
        return

    def get_statement_sequence(self):
        return

