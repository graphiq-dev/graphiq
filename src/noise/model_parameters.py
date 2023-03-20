import src.circuit.ops as ops


def equal_time_dict(ops_list, duration):
    """
    A helper function. Creates a dictionary of operation types and assigns them all an equal time duration.
    :param ops_list: list of operations
    :type ops_list: list
    :param duration: a time duration
    :type duration: int or float
    :return: the gate duration dictionary for the given operations
    :rtype: dict
    """
    duration_dict = {}
    ops_types = [type(op) for op in ops_list]
    assert isinstance(
        duration, (float, int)
    ), "time duration must be a float or integer number"
    for types in ops_types:
        duration_dict[types] = duration
    return duration_dict


gate_duration_dict = {
    ops.CNOT: 1,
    ops.CZ: 1,
    ops.SigmaX: 1,
    ops.SigmaY: 1,
    ops.SigmaZ: 1,
    ops.Hadamard: 1,
    ops.Phase: 1,
    ops.ClassicalCNOT: 1,
    ops.ClassicalCZ: 1,
    ops.MeasurementZ: 1,
    ops.MeasurementCNOTandReset: 1,
    ops.ClassicalControlledPairOperationBase: 1,
    ops.ControlledPairOperationBase: 1,
    ops.Identity: 1,
    ops.Input: 0,
    ops.Output: 0,
}
noise_parameters = {
    "error_rate": 0.05,
    "cut_off_prob": 0.99,
    "criteria": "reg_as_control",
    "noise_type": ["x"],
    "reg_specific_noise": {},  # {'regtype-reg string':'noise type', etc.} e.g. {'e0': 'x', 'e1': ['y', 'z']}
}  # having a list as noise type would apply both kinds of noises separately to the register

error_ops = {"x": ops.SigmaX, "y": ops.SigmaY, "z": ops.SigmaZ, "I": ops.Identity}
