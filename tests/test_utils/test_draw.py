import pytest
from src.utils.draw import Columns
from src.utils.draw import Painter


# Test Columns class
# TODO: Use pytest.parameterize
def test_columns_init():
    columns = Columns(col_num=1)

    assert columns.size == 1
    assert columns.columns == [[0]]
    assert columns.col_width == [0]


def test_add_new_column():
    columns = Columns(col_num=1, size=3)
    columns.add_new_column()

    assert columns.size == 3
    assert columns.columns == [[0, 0, 0], [0, 0, 0]]
    assert columns.col_width == [0, 0]


def test_expand_cols():
    columns = Columns(col_num=3, size=3)
    columns.expand_cols(size=2)

    assert columns.size == 5
    assert columns.columns == [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
    assert columns.col_width == [0, 0, 0]


def test_find_and_add_to_empty_col():
    columns = Columns(col_num=3, size=3)
    columns.find_and_add_to_empty_col(from_index=0, to_index=2)

    assert columns.size == 3
    assert columns.columns == [[1, 1, 0], [0, 0, 0], [0, 0, 0]]


# Test Painter class
def test_painter_init():
    painter = Painter()

    assert painter.next_reg_position == 50

    assert painter.registers_position == {
        "qreg": {},
        "creg": {},
    }
    assert painter.registers_mapping == []

    assert painter.gates == []
    assert painter.measurements == []
    assert painter.barriers == []
    assert painter.resets == []
    assert painter.classical_controls == []


@pytest.mark.parametrize(
    "reg_name, size, reg_type, reg_mapping, next_pos",
    [
        ("q", 4, "qreg", ["q0", "q1", "q2", "q3"], 250),
        ("test", 4, "qreg", ["test0", "test1", "test2", "test3"], 250),
        ("c", 4, "creg", ["c4"], 100),
    ],
)
def test_painter_add_regiter(reg_name, size, reg_type, reg_mapping, next_pos):
    painter = Painter()
    painter.add_register(reg_name=reg_name, size=size, reg_type=reg_type)

    assert painter.registers_mapping == reg_mapping
    assert painter.next_reg_position == next_pos


def test_painter_add_register_error():
    painter = Painter()
    with pytest.raises(ValueError, match="Register type must be creg or qreg"):
        painter.add_register("q", 4, "test")


@pytest.mark.parametrize(
    "gate_name, qargs, params, controls, expected_output",
    [
        (
            "H",
            ["p1"],
            {},
            [],
            {"gate_name": "H", "params": {}, "qargs": ["p1"], "controls": [], "col": 0},
        ),
        (
            "CX",
            ["p1"],
            {},
            ["p2"],
            {
                "gate_name": "CX",
                "params": {},
                "qargs": ["p1"],
                "controls": ["p2"],
                "col": 0,
            },
        ),
        (
            "RX",
            ["p1"],
            {"theta": "pi/2"},
            [],
            {
                "gate_name": "RX",
                "params": {"theta": "pi/2"},
                "qargs": ["p1"],
                "controls": [],
                "col": 0,
            },
        ),
    ],
)
def test_painter_add_gates(gate_name, qargs, params, controls, expected_output):
    painter = Painter()
    painter.add_register("p", 4)

    gate_info = painter.add_gate(gate_name, qargs, params, controls)
    assert gate_info == expected_output


def test_painter_add_gate_error():
    painter = Painter()
    painter.add_register("p", 4)

    with pytest.raises(ValueError, match="Multi-control gate is not supported yet."):
        gate_info = painter.add_gate("test", ["p1"], {}, ["p0", "p2"])


def test_painter_add_measurement():
    painter = Painter()
    painter.add_register("p", 4)
    painter.add_register("c", 4, "creg")

    measurement_info = painter.add_measurement("p1", "c4")
    assert measurement_info == {"col": 0, "qreg": "p1", "creg": "c4", "cbit": 0}


def test_painter_add_barrier():
    painter = Painter()
    painter.add_register("p", 4)
    painter.add_register("c", 4, "creg")

    barrier_info = painter.add_barrier("p1")
    assert barrier_info == {"col": 0, "qreg": "p1"}


def test_painter_add_reset():
    painter = Painter()
    painter.add_register("p", 4)
    painter.add_register("c", 4, "creg")

    reset_info = painter.add_reset("p1")
    assert reset_info == {"col": 0, "qreg": "p1"}


def test_painter_add_classical_control():
    painter = Painter()
    painter.add_register("p", 4)
    painter.add_register("c", 4, "creg")

    classical_control_info = painter.add_classical_control(
        creg="c4", gate_name="X", qargs=["p1"]
    )
    assert classical_control_info == {
        "col": 0,
        "creg": "c4",
        "gate_info": {"gate_name": "X", "params": {}, "qargs": ["p1"], "control": []},
    }
