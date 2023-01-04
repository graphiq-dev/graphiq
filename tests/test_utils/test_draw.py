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
# def test_painter_init():
#     painter = Painter()
#
#     pass
