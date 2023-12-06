from graphiq.backends.stabilizer.state import *


def test_state1():
    state = Stabilizer(4)
    print(state.tableau.destabilizer_to_labels())
    print(state.tableau.stabilizer_to_labels())
    state.tableau.stabilizer_z = np.array(
        [[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]]
    )
    state.tableau.destabilizer_z = np.array(
        [[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]]
    )
    state.apply_hadamard(1)
    state.tableau.destabilizer_from_labels(["XYXI", "ZYZI", "YIII", "ZZZZ"])
    print(state.tableau.destabilizer)
