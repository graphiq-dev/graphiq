class DrawingManager:
    def __init__(self, openqasm: str):
        self.openqasm = openqasm

        # TODO: Need to calculate width and height
        self.position = {
            "label": {
                "width": 70,
                "height": 350,
            },
            "detail": {
                "width": 3000,
                "height": 350,
            }
        }

        def calculate_height(n_quantum_reg, n_classical_reg):
            height = 50

            height += 50 * n_quantum_reg
            height += 50 * n_classical_reg

            return height

