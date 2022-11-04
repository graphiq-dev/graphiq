import numpy as np
import numpy.testing as test
import pytest
import random

from src.circuit import Register
from tests.test_flags import visualization


@pytest.mark.parametrize(
    "reg_dict, is_multi_qubit", [({"e": [], "p": []}, False),
                                 ({"e": [1, 1], "p": [1, 1, 1]}, False),
                                 ({"e": [1, 1], "p": [1, 2, 3]}, True)]
)
def test_register_initialization(reg_dict, is_multi_qubit):
    register = Register(reg_dict=reg_dict, is_multi_qubit=is_multi_qubit)

    assert register.register == reg_dict
    assert register.is_multi_qubit is is_multi_qubit


@pytest.mark.parametrize(
    "reg_dict, is_multi_qubit, error_message", [
        (None, False, "Register dict can not be None or empty"),
        ({}, False, "Register dict can not be None or empty"),
        ({"e": [1, "test"], "p": [1, 1, 1]}, False, "The input data contains non-numerical value"),
        ({"e": [1, 2], "p": [1, 2, 3]}, False, "Register is not multi-qubit register but has value more than 1"),
    ]
)
def test_register_initialization_wrong_input(reg_dict, is_multi_qubit, error_message):
    with pytest.raises(ValueError, match=error_message):
        register = Register(reg_dict=reg_dict, is_multi_qubit=is_multi_qubit)


def test_get_set_item():
    register = Register({"e": [1, 1], "p": [1, 1, 1]}, False)

    # test get
    assert register["e"] == [1, 1]
    assert register["p"] == [1, 1, 1]

    # test set
    register["e"] = [1, 1, 1, 1, 1]
    register["p"] = []
    register["c"] = [1]

    assert register["e"] == [1, 1, 1, 1, 1]
    assert register["p"] == []
    assert register["c"] == [1]


@pytest.mark.parametrize(
    "reg_dict, error_message", [
        ({"e": [1, "test"], "p": [1, 1, 1]}, "The input data contains non-numerical value"),
        ({"e": [1, 2], "p": [1, 1, 1]}, "The register only supports single-qubit registers"),
    ]
)
def test_set_item_wrong_input(reg_dict, error_message):
    register = Register(reg_dict={"e": [1, 1], "p": [1, 1, 1]})

    with pytest.raises(ValueError, match=error_message):
        register["e"] = reg_dict["e"]


def test_n_quantum():
    register = Register(reg_dict={"e": [1, 1], "p": [1, 1, 1], "c": [1]})
    assert register.n_quantum == 5


def test_add_register():
    register = Register(reg_dict={"e": [1, 1], "p": [1, 1, 1], "c": [1]})
    register.add_register(reg_type="e", size=1)
    assert register["e"] == [1, 1, 1]

    with pytest.raises(ValueError):
        register.add_register(reg_type="test")
    with pytest.raises(ValueError):
        register.add_register(reg_type="e", size=0)
    with pytest.raises(ValueError):
        register.add_register(reg_type="e", size=2)

    # test add with multi-qubit register
    register = Register(reg_dict={"e": [1, 1], "p": [1, 1, 1], "c": [1]}, is_multi_qubit=True)
    register.add_register(reg_type="e", size=2)
    assert register["e"] == [1, 1, 2]


def test_expand_register():
    register = Register(reg_dict={"e": [1, 1], "p": [1, 1, 1], "c": [1]})

    with pytest.raises(ValueError):
        register.expand_register(reg_type="test", register=0, new_size=2)
    with pytest.raises(ValueError):
        register.expand_register(reg_type="e", register=0, new_size=2)

    # test expand with multi-qubit register
    register = Register(reg_dict={"e": [1, 1], "p": [1, 1, 1], "c": [1]}, is_multi_qubit=True)
    register.expand_register(reg_type="e", register=0, new_size=3)
    assert register["e"] == [3, 1]
    with pytest.raises(ValueError):
        register.expand_register(reg_type="e", register=0, new_size=2)


def test_next_register():
    register = Register(reg_dict={"e": [1, 1], "p": [1, 1, 1, 1, 1], "c": [1]})
    assert register.next_register(reg_type="p", register=1) == 1

    with pytest.raises(ValueError):
        register.next_register(reg_type="test", register=0)

