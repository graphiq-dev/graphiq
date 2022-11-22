import src.ops as ops
from src.circuit import *


def circuit_to_json(circuit: CircuitBase):
    json_dict = {
        "register": circuit.register,
        "register_depth": circuit.register_depth
    }
    sequence = []

    for op in circuit.sequence():
        gate_info = {}
        if isinstance(op, ops.OneQubitOperationBase):
            gate_info["type"] = "1q"
            gate_info["register"] = op.register
            gate_info["reg_type"] = op.reg_type
        if isinstance(op, ops.ControlledPairOperationBase):
            gate_info["type"] = "cp"
            gate_info["control"] = op.control
            gate_info["control_type"] = op.control_type
            gate_info["target"] = op.target
            gate_info["target_type"] = op.target_type
        if isinstance(op, ops.ClassicalControlledPairOperationBase):
            gate_info["type"] = "ccp"
        if gate_info:
            sequence.append(gate_info)

    json_dict["sequence"] = sequence

    json_obj = json.dumps(json_dict, indent=3)
    return json_obj

