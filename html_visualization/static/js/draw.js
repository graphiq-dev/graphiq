// this file is used to draw on html
function draw_register_info(registers) {
    qreg = registers["qreg"]
    creg = registers["creg"]

    for (reg in qreg) {
        draw_register_label(reg, 50, qreg[reg])
        draw_quantum_register(60, qreg[reg])
    }

    for (reg in creg) {
        draw_register_label(reg, 50, creg[reg])
        draw_classical_register(60, creg[reg])
    }
}

function draw_classical_control(classical_control) {
    creg_pos = visualization_info.registers.creg[classical_control.creg]
    qreg_pos = visualization_info.registers.qreg[classical_control.gate_info.qargs]

    gate = one_qubit_gate(
        classical_control.x_pos,
        qreg_pos,
        classical_control.gate_info.gate_name,
        classical_control.gate_info.qargs,
        classical_control.gate_info.params,
    )
    create_classical_control_at(gate, classical_control.x_pos, qreg_pos+20, creg_pos)
}

function draw_gate(gate) {
    register = visualization_info.registers.qreg[gate.qargs]
    if (gate.gate_name === "CX") {
        y2 = visualization_info.registers.qreg[gate.controls[0]] - register
        cnot(gate.x_pos, register, y2)
    }
    else if (gate.gate_name == 'CZ') {
        y2 = visualization_info.registers.qreg[gate.controls[0]] - register
        cz(gates[i].x_pos, register, y2)
    }
    else {
        g = one_qubit_gate(gate.x_pos, register, gate.gate_name, gate.qargs, gate.params, gate.controls)
    }
}

function draw(ops) {
    for (let i = 0; i < ops.length; i++) {
        console.log(ops[i])

        if (ops[i].type == 'gate') {
            draw_gate(ops[i])
        }
        if (ops[i].type == 'barrier') {
            qreg_pos =  visualization_info.registers.qreg[ops[i].qreg]
            g = barrier(ops[i].x_pos, qreg_pos)
        }
        if (ops[i].type == 'reset') {
            qreg_pos =  visualization_info.registers.qreg[ops[i].qreg]
            g = reset(ops[i].x_pos, qreg_pos, ops[i].qreg)
        }
        if (ops[i].type == 'measure') {
            qreg_pos =  visualization_info.registers.qreg[ops[i].qreg]
            creg_pos =  visualization_info.registers.creg[ops[i].creg]
            g = measure(ops[i].x_pos, qreg_pos, creg_pos)
        }
        if (ops[i].type == 'if') {
            draw_classical_control(ops[i])
        }
    }
}