// this file is used to draw on html

function draw_register_info(registers) {
    qreg = registers["qreg"]
    creg = registers["creg"]

    for (reg in qreg) {
        draw_register_label(reg, 70, qreg[reg])
        draw_quantum_register(0, qreg[reg])
    }

    for (reg in creg) {
        draw_register_label(reg, 70, creg[reg])
        draw_classical_register(0, creg[reg])
    }
}

function draw_qreg(qreg) {
    for (reg in qreg) {
        draw_register_label(reg, 70, qreg[reg])
        draw_quantum_register(0, qreg[reg])
    }
}

function draw_creg(creg) {
    for (reg in creg) {
        draw_register_label(reg, 70, creg[reg])
        draw_quantum_register(0, creg[reg])
    }
}

function draw_gate_info(gates) {
    for (let i = 0; i < gates.length; i++) {
        register = visualization_info.registers.qreg[gates[i].qargs]

        if (gates[i].gate_name === "CX") {
            y2 = visualization_info.registers.qreg[gates[i].controls] - register
            cnot(gates[i].x_pos, register, y2)
        }
        else if (gates[i].gate_name === "CZ") {
            y2 = visualization_info.registers.qreg[gates[i].controls] - register
            cz(gates[i].x_pos, register, y2)
        }
        else {
            g = one_qubit_gate(gates[i].x_pos, register, gates[i].gate_name, gates[i].qargs, gates[i].params)

            if (gates[i].controls.length !== 0) {
                color = g.select("rect").style("fill")

                x = g.select("rect").attr("width") / 2
                y1 = g.select("rect").attr("height")
                y2 = visualization_info.registers.qreg[gates[i].controls] - register + 20
                create_control_at(g, x, y1, y2, color)
            }
        }
    }
}

function draw_measurement(measurements) {
    for (let i = 0; i < measurements.length; i++) {
        qreg_pos =  visualization_info.registers.qreg[measurements[i].qreg]
        creg_pos =  visualization_info.registers.creg[measurements[i].creg]
        creg_pos = creg_pos - qreg_pos

        // console.log(measurements[i])
        g = measure(measurements[i].x_pos, qreg_pos, creg_pos)
    }
}

function draw_resets(resets) {
    for (let i = 0; i < resets.length; i++) {
        qreg_pos =  visualization_info.registers.qreg[resets[i].qreg]

        // console.log(measurements[i])
        g = reset(resets[i].x_pos, qreg_pos, resets[i].qreg)
    }
}

function draw_barriers(barriers) {
    for (let i = 0; i < barriers.length; i++) {
        qreg_pos =  visualization_info.registers.qreg[barriers[i].qreg]

        // console.log(measurements[i])
        g = barrier(barriers[i].x_pos, qreg_pos, barriers[i].qreg)
    }
}