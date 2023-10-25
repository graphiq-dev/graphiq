import pytest
from src.utils.draw import Columns
from src.utils.draw import Painter
from benchmarks.circuits import *


# Test Columns class
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


def test_set_all_col_error():
    columns = Columns(col_num=3, size=3)
    columns.expand_cols(size=2)

    with pytest.raises(ValueError, match="Index parameter must be an integer"):
        columns.set_all_col_element(index="test")


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
    assert painter.ops == []


def test_painter_init_with_gate_mapping():
    painter = Painter(gate_mapping={"CX": 10})
    assert painter.gate_mapping == {"CX": 10}


@pytest.mark.parametrize(
    "reg_name, size, reg_type, reg_mapping, next_pos",
    [
        ("q", 4, "qreg", ["q[0]", "q[1]", "q[2]", "q[3]"], 250),
        ("test", 4, "qreg", ["test[0]", "test[1]", "test[2]", "test[3]"], 250),
        ("c", 4, "creg", ["c[4]"], 100),
    ],
)
def test_painter_add_register(reg_name, size, reg_type, reg_mapping, next_pos):
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
            ["p[1]"],
            {},
            [],
            {
                "type": "gate",
                "gate_name": "H",
                "params": {},
                "qargs": ["p[1]"],
                "controls": [],
                "col": 0,
            },
        ),
        (
            "CX",
            ["p[1]"],
            {},
            ["p[2]"],
            {
                "type": "gate",
                "gate_name": "CX",
                "params": {},
                "qargs": ["p[1]"],
                "controls": ["p[2]"],
                "col": 0,
            },
        ),
        (
            "RX",
            ["p[1]"],
            {"theta": "pi/2"},
            [],
            {
                "type": "gate",
                "gate_name": "RX",
                "params": {"theta": "pi/2"},
                "qargs": ["p[1]"],
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

    with pytest.raises(
        ValueError, match="Gate that act on multi-qargs is not supported yet."
    ):
        gate_info = painter.add_gate("test", ["p[1]", "p[2]"])


def test_painter_add_measurement():
    painter = Painter()
    painter.add_register("p", 4)
    painter.add_register("c", 4, "creg")

    measurement_info = painter.add_measurement("p[1]", "c[4]")
    assert measurement_info == {
        "type": "measure",
        "col": 0,
        "qreg": "p[1]",
        "creg": "c[4]",
        "cbit": 0,
    }


def test_painter_add_barrier():
    painter = Painter()
    painter.add_register("p", 4)
    painter.add_register("c", 4, "creg")

    painter.add_barriers(["p[1]"])
    assert painter.ops[-1] == {"type": "barrier", "col": 0, "qreg": "p[1]"}


def test_painter_add_reset():
    painter = Painter()
    painter.add_register("p", 4)
    painter.add_register("c", 4, "creg")

    reset_info = painter.add_reset("p[1]")
    assert reset_info == {"type": "reset", "col": 0, "qreg": "p[1]"}


def test_painter_add_classical_control():
    painter = Painter()
    painter.add_register("p", 4)
    painter.add_register("c", 4, "creg")

    classical_control_info = painter.add_classical_control(
        creg="c[4]", gate_name="X", qargs=["p[1]"]
    )
    assert classical_control_info == {
        "type": "if",
        "col": 0,
        "creg": "c[4]",
        "gate_info": {"gate_name": "X", "params": {}, "qargs": ["p[1]"], "control": []},
    }


def test_add_classical_error():
    painter = Painter()
    painter.add_register("p", 4)
    painter.add_register("c", 4, "creg")

    with pytest.raises(
        ValueError,
        match="Multiple qubits gate is not supported in classical control right now",
    ):
        classical_control_info = painter.add_classical_control(
            creg="c[4]", gate_name="X", qargs=["p[1]", "p[2]"]
        )


def test_build_visualization_info():
    painter = Painter()
    painter.add_register("p", 4)
    painter.add_register("c", 4, "creg")

    painter.add_gate("H", ["p[1]"])
    painter.add_gate("CX", ["p[0]"], controls=["p[1]"])
    painter.add_barriers(["p[0]", "p[1]", "p[2]", "p[3]"])

    info = painter.build_visualization_info()
    assert info == {
        "width": 340.0,
        "registers": {
            "qreg": {"p[0]": 50, "p[1]": 100, "p[2]": 150, "p[3]": 200},
            "creg": {"c[4]": 250},
        },
        "ops": [
            {
                "type": "gate",
                "gate_name": "H",
                "params": {},
                "qargs": ["p[1]"],
                "controls": [],
                "col": 0,
                "x_pos": 120.0,
            },
            {
                "type": "gate",
                "gate_name": "CX",
                "params": {},
                "qargs": ["p[0]"],
                "controls": ["p[1]"],
                "col": 1,
                "x_pos": 180.0,
            },
            {"type": "barrier", "col": 2, "qreg": "p[0]", "x_pos": 240.0},
            {"type": "barrier", "col": 2, "qreg": "p[1]", "x_pos": 240.0},
            {"type": "barrier", "col": 2, "qreg": "p[2]", "x_pos": 240.0},
            {"type": "barrier", "col": 2, "qreg": "p[3]", "x_pos": 240.0},
        ],
    }


def test_load_openqasm_0():
    circuit, state = ghz3_state_circuit()
    openqasm_str = circuit.to_openqasm()

    painter = Painter()
    painter.load_openqasm_str(openqasm_str)

    assert painter.registers_mapping == ["p0[0]", "p1[0]", "p2[0]", "e0[0]", "c0[1]"]
    assert painter.ops == [
        {
            "type": "gate",
            "gate_name": "H",
            "params": {},
            "qargs": ["e0[0]"],
            "controls": [],
            "col": 0,
        },
        {
            "type": "gate",
            "gate_name": "CX",
            "params": {},
            "qargs": ["p0[0]"],
            "controls": ["e0[0]"],
            "col": 1,
        },
        {
            "type": "gate",
            "gate_name": "CX",
            "params": {},
            "qargs": ["p1[0]"],
            "controls": ["e0[0]"],
            "col": 2,
        },
        {
            "type": "gate",
            "gate_name": "CX",
            "params": {},
            "qargs": ["p2[0]"],
            "controls": ["e0[0]"],
            "col": 3,
        },
        {
            "type": "gate",
            "gate_name": "H",
            "params": {},
            "qargs": ["p2[0]"],
            "controls": [],
            "col": 4,
        },
        {
            "type": "gate",
            "gate_name": "H",
            "params": {},
            "qargs": ["e0[0]"],
            "controls": [],
            "col": 4,
        },
        {"type": "barrier", "col": 5, "qreg": "p0[0]"},
        {"type": "barrier", "col": 5, "qreg": "p1[0]"},
        {"type": "barrier", "col": 5, "qreg": "p2[0]"},
        {"type": "barrier", "col": 5, "qreg": "e0[0]"},
        {"type": "measure", "col": 6, "qreg": "e0[0]", "creg": "c0[1]", "cbit": 0},
        {
            "type": "if",
            "col": 7,
            "creg": "c0[1]",
            "gate_info": {
                "gate_name": "X",
                "params": {},
                "qargs": ["p2[0]"],
                "control": [],
            },
        },
        {"type": "barrier", "col": 8, "qreg": "e0[0]"},
        {"type": "barrier", "col": 8, "qreg": "p2[0]"},
        {"type": "reset", "col": 9, "qreg": "e0[0]"},
        {"type": "barrier", "col": 10, "qreg": "p0[0]"},
        {"type": "barrier", "col": 10, "qreg": "p1[0]"},
        {"type": "barrier", "col": 10, "qreg": "p2[0]"},
        {"type": "barrier", "col": 10, "qreg": "e0[0]"},
        {
            "type": "gate",
            "gate_name": "H",
            "params": {},
            "qargs": ["p2[0]"],
            "controls": [],
            "col": 11,
        },
    ]


def test_load_openqasm_1():
    openqasm_str = """
    OPENQASM 2.0;
    
    qreg p[4];
    creg c[4];
    
    reset p;
    measure p -> c;
    """

    painter = Painter()
    painter.load_openqasm_str(openqasm_str)

    assert painter.registers_mapping == ["p[0]", "p[1]", "p[2]", "p[3]", "c[4]"]
    assert painter.ops == [
        {"type": "reset", "col": 0, "qreg": "p[0]"},
        {"type": "reset", "col": 0, "qreg": "p[1]"},
        {"type": "reset", "col": 0, "qreg": "p[2]"},
        {"type": "reset", "col": 0, "qreg": "p[3]"},
        {"type": "measure", "col": 1, "qreg": "p[0]", "creg": "c[4]", "cbit": 0},
        {"type": "measure", "col": 2, "qreg": "p[1]", "creg": "c[4]", "cbit": 1},
        {"type": "measure", "col": 3, "qreg": "p[2]", "creg": "c[4]", "cbit": 2},
        {"type": "measure", "col": 4, "qreg": "p[3]", "creg": "c[4]", "cbit": 3},
    ]
