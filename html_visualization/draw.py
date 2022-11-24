class DrawingManager:
    def __init__(self, openqasm: str):
        self.openqasm = openqasm
        self.position = {
            "register": {}
        }

